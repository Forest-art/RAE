"""
S-RAE: Semantic-Guided Regularized Autoencoder

A unified model for semantic understanding and image generation.
- Teacher: Frozen DINOv2 for semantic guidance
- Student: Trainable MAE-style encoder with continuous bottleneck
- Decoder: Lightweight transformer for image reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from transformers import AutoImageProcessor, Dinov2Model
try:
    from transformers import Dinov2WithRegistersModel
except Exception:  # pragma: no cover - older transformers versions
    Dinov2WithRegistersModel = None
from .decoders import GeneralDecoder
from .decoders.utils import ViTMAEConfig


class SRAE(nn.Module):
    """
    Semantic-Guided Regularized Autoencoder (S-RAE)
    
    Combines DINOv2 semantic understanding with MAE-style generation
    through a continuous latent space with semantic alignment.
    """
    
    def __init__(
        self,
        # Teacher/Student backbone config
        dinov2_model_name: str = "facebook/dinov2-with-registers-base",
        img_size: int = 256,
        encoder_input_size: int = 224,
        patch_size: Optional[int] = 14,  # encoder patch size; inferred from backbone if None
        
        # Masking config
        mask_ratio: float = 0.75,
        
        # Bottleneck config
        bottleneck_dim: Optional[int] = None,  # None means same as encoder_dim
        use_l2_norm: bool = True,
        
        # Projector config (for alignment)
        projector_hidden_dim: int = 4096,
        
        # Decoder config
        decoder_num_layers: int = 8,
        decoder_num_heads: int = 16,
        decoder_dim: int = 512,
        
        # Loss weights
        loss_rec_weight: float = 1.0,
        loss_align_weight: float = 0.1,
        loss_reg_weight: float = 0.01,
        
        # Teacher output type
        teacher_output_type: str = "cls",  # "cls" or "patch"
    ):
        super().__init__()
        
        self.img_size = int(img_size)
        self.encoder_input_size = int(encoder_input_size)
        self.mask_ratio = mask_ratio
        self.loss_rec_weight = loss_rec_weight
        self.loss_align_weight = loss_align_weight
        self.loss_reg_weight = loss_reg_weight
        self.teacher_output_type = teacher_output_type
        
        # ========== Teacher (Frozen DINOv2) ==========
        print(f"Loading Teacher DINOv2 from {dinov2_model_name}...")
        self.teacher = self._load_backbone(dinov2_model_name)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        teacher_dim = self.teacher.config.hidden_size
        print(f"Teacher loaded. Hidden dim: {teacher_dim}")
        
        # ========== Student (Trainable DINOv2) ==========
        print(f"Loading Student DINOv2 from {dinov2_model_name}...")
        self.student_encoder = self._load_backbone(dinov2_model_name)
        student_dim = self.student_encoder.config.hidden_size
        print(f"Student loaded. Hidden dim: {student_dim}")

        self.teacher_prefix_tokens = 1 + int(getattr(self.teacher.config, "num_register_tokens", 0))
        self.student_prefix_tokens = 1 + int(getattr(self.student_encoder.config, "num_register_tokens", 0))

        # RAE-aligned encoder preprocessing (resize + mean/std normalization).
        processor = self._load_image_processor(dinov2_model_name)
        encoder_mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        encoder_std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("encoder_mean", encoder_mean, persistent=False)
        self.register_buffer("encoder_std", encoder_std, persistent=False)

        inferred_encoder_patch = int(getattr(self.student_encoder.config, "patch_size", 14))
        if patch_size is not None and int(patch_size) != inferred_encoder_patch:
            raise ValueError(
                f"Configured encoder patch_size={patch_size} does not match student backbone patch_size={inferred_encoder_patch}."
            )
        self.encoder_patch_size = inferred_encoder_patch

        if self.encoder_input_size % self.encoder_patch_size != 0:
            raise ValueError(
                f"encoder_input_size ({self.encoder_input_size}) must be divisible by encoder patch_size ({self.encoder_patch_size})."
            )
        self.encoder_grid_size = self.encoder_input_size // self.encoder_patch_size
        self.num_patches = self.encoder_grid_size * self.encoder_grid_size

        if self.img_size % self.encoder_grid_size != 0:
            raise ValueError(
                f"img_size ({self.img_size}) must be divisible by encoder token grid size ({self.encoder_grid_size})."
            )
        # Keep the same token grid as encoder; decoder patch size adapts to reconstruction resolution.
        self.decoder_patch_size = self.img_size // self.encoder_grid_size
        
        # ========== Bottleneck ==========
        bottleneck_dim = bottleneck_dim or student_dim
        self.bottleneck_dim = bottleneck_dim
        
        # Linear projection + normalization
        self.bottleneck_proj = nn.Linear(student_dim, bottleneck_dim)
        self.bottleneck_norm = nn.LayerNorm(bottleneck_dim) if not use_l2_norm else None
        self.use_l2_norm = use_l2_norm
        
        # ========== Projector (for alignment loss only) ==========
        self.projector = nn.Sequential(
            nn.Linear(bottleneck_dim, projector_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(projector_hidden_dim),
            nn.Linear(projector_hidden_dim, teacher_dim),
        )
        
        # ========== Decoder (MAE-style) ==========
        # Create decoder config
        decoder_config = ViTMAEConfig(
            hidden_size=decoder_dim,
            num_hidden_layers=decoder_num_layers,
            num_attention_heads=decoder_num_heads,
            intermediate_size=decoder_dim * 4,
            image_size=self.img_size,
            patch_size=self.decoder_patch_size,
            num_channels=3,
            decoder_hidden_size=decoder_dim,
            decoder_num_hidden_layers=decoder_num_layers,
            decoder_num_attention_heads=decoder_num_heads,
            decoder_intermediate_size=decoder_dim * 4,
        )
        
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.num_patches)
        
        # Decoder embed: projects bottleneck to decoder dim
        self.decoder_embed = nn.Linear(bottleneck_dim, decoder_dim)
        
        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        print(f"S-RAE initialized:")
        print(f"  - Encoder input size: {self.encoder_input_size}")
        print(f"  - Reconstruction size: {self.img_size}")
        print(f"  - Encoder patch size: {self.encoder_patch_size}")
        print(f"  - Decoder patch size: {self.decoder_patch_size}")
        print(f"  - Num patches: {self.num_patches}")
        print(f"  - Bottleneck dim: {bottleneck_dim}")
        print(f"  - Decoder dim: {decoder_dim}")
        print(f"  - Mask ratio: {mask_ratio}")

    @staticmethod
    def _load_image_processor(model_name: str):
        load_errors = []
        for local_only in (True, False):
            try:
                return AutoImageProcessor.from_pretrained(
                    model_name,
                    local_files_only=local_only,
                )
            except Exception as exc:
                mode = "local-only" if local_only else "remote-enabled"
                load_errors.append(f"{mode}: {exc}")
        raise RuntimeError(
            f"Failed to load AutoImageProcessor for '{model_name}'. Errors: {' | '.join(load_errors)}"
        )

    @staticmethod
    def _load_backbone(model_name: str):
        candidate_classes = []
        if Dinov2WithRegistersModel is not None:
            candidate_classes.append(Dinov2WithRegistersModel)
        candidate_classes.append(Dinov2Model)

        load_errors = []
        for model_cls in candidate_classes:
            for local_only in (True, False):
                try:
                    return model_cls.from_pretrained(
                        model_name,
                        local_files_only=local_only,
                    )
                except Exception as exc:
                    mode = "local-only" if local_only else "remote-enabled"
                    load_errors.append(f"{model_cls.__name__}({mode}): {exc}")

        raise RuntimeError(
            f"Failed to load DINOv2 backbone '{model_name}'. Errors: {' | '.join(load_errors)}"
        )

    def _preprocess_for_encoder(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.encoder_input_size, self.encoder_input_size):
            x = F.interpolate(
                x,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic",
                align_corners=False,
            )
        mean = self.encoder_mean.to(device=x.device, dtype=x.dtype)
        std = self.encoder_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def _get_student_register_tokens(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
        if self.student_prefix_tokens <= 1:
            return None

        register_tokens = getattr(self.student_encoder.embeddings, "register_tokens", None)
        if register_tokens is None:
            register_tokens = getattr(self.student_encoder.embeddings, "register_token", None)
        if register_tokens is None:
            return None

        if register_tokens.dim() == 2:
            register_tokens = register_tokens.unsqueeze(0)
        return register_tokens.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        
    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Random masking of patches.
        Args:
            x: [B, N, D] patch embeddings
            mask_ratio: ratio of patches to mask
        Returns:
            x_visible: [B, N_visible, D] visible patches
            mask: [B, N] binary mask (1 = masked, 0 = visible)
            ids_restore: [B, N] indices to restore original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        # Random permutation
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep visible patches
        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # Generate mask (1 = masked, 0 = visible)
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_visible, mask, ids_restore
    
    @torch.no_grad()
    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through frozen teacher"""
        x = self._preprocess_for_encoder(x)
        
        outputs = self.teacher(pixel_values=x, output_hidden_states=True)
        
        if self.teacher_output_type == "cls":
            # Use CLS token
            return outputs.last_hidden_state[:, 0]  # [B, D]

        # Use mean of patch tokens
        return outputs.last_hidden_state[:, self.teacher_prefix_tokens:].mean(dim=1)  # [B, D]
    
    def forward_student(self, x: torch.Tensor, mask_ratio: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward through student encoder (with masking)
        Returns:
            z: bottleneck latent [B, N_visible, bottleneck_dim]
            z_proj: projected latent for alignment [B, N_visible, teacher_dim]
            mask: [B, N]
            ids_restore: [B, N]
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        x = self._preprocess_for_encoder(x)
        
        # Get patch embeddings from student's embedding layer
        # DINOv2 uses a patch embedding in the model
        patch_embeds = self.student_encoder.embeddings.patch_embeddings.projection(x)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # Random masking
        x_visible, mask, ids_restore = self.random_masking(patch_embeds, mask_ratio)
        
        # Add position embeddings (only for visible patches)
        # DINOv2 uses interpolated position embeddings
        B, N_visible, D = x_visible.shape
        
        # Get position embeddings for visible patches
        num_patches = patch_embeds.shape[1]
        pos_embed = self.student_encoder.embeddings.position_embeddings  # [1, N+1, D]
        
        # Interpolate position embeddings if needed
        if pos_embed.shape[1] - 1 != num_patches:
            pos_embed = self.student_encoder.embeddings.interpolate_pos_encoding(
                pos_embed, self.encoder_input_size, self.encoder_input_size
            )
        
        # Extract position embeddings for visible patches
        # Note: pos_embed includes CLS token at index 0
        pos_embed_visible = pos_embed[:, 1:].expand(B, -1, -1)  # [B, N, D]
        # Gather for visible patches only
        ids_keep = torch.argsort(ids_restore, dim=1)[:, :N_visible]
        pos_embed_visible = torch.gather(pos_embed_visible, dim=1, 
                                         index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        x_visible = x_visible + pos_embed_visible
        
        # Add CLS token
        cls_token = self.student_encoder.embeddings.cls_token.expand(B, -1, -1)
        cls_token = cls_token + pos_embed[:, :1]
        register_tokens = self._get_student_register_tokens(B, cls_token.dtype, cls_token.device)
        if register_tokens is not None:
            x_full = torch.cat([cls_token, register_tokens, x_visible], dim=1)
            prefix_tokens = 1 + register_tokens.shape[1]
        else:
            x_full = torch.cat([cls_token, x_visible], dim=1)  # [B, 1+N_visible, D]
            prefix_tokens = 1
        
        # Pass through transformer
        hidden_states = self.student_encoder.encoder(x_full).last_hidden_state
        
        # Extract patch tokens (exclude cls/register tokens)
        patch_tokens = hidden_states[:, prefix_tokens:]  # [B, N_visible, D]
        
        # Bottleneck
        z = self.bottleneck_proj(patch_tokens)  # [B, N_visible, bottleneck_dim]
        
        if self.use_l2_norm:
            z = F.normalize(z, p=2, dim=-1)
        else:
            z = self.bottleneck_norm(z)
        
        # Projector (for alignment)
        z_proj = self.projector(z)  # [B, N_visible, teacher_dim]
        
        return z, z_proj, mask, ids_restore
    
    def forward_decoder(self, z: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Decode from bottleneck latent.
        Args:
            z: [B, N_visible, bottleneck_dim]
            ids_restore: [B, N]
        Returns:
            logits: [B, N, patch_dim] reconstruction logits
        """
        B, N_visible, _ = z.shape
        N = self.num_patches
        
        # Embed to decoder dimension
        x = self.decoder_embed(z)  # [B, N_visible, decoder_dim]
        
        # Append mask tokens for masked patches
        mask_tokens = self.mask_token.expand(B, N - N_visible, -1)
        x_full = torch.cat([x, mask_tokens], dim=1)  # [B, N, decoder_dim]
        
        # Unshuffle to original order
        x_full = torch.gather(
            x_full,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.size(-1)),
        )
        
        # Pass through decoder. GeneralDecoder handles CLS token and positional embeddings.
        outputs = self.decoder(x_full, drop_cls_token=False)
        logits = outputs.logits  # [B, N, patch_dim]
        
        return logits
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patch logits to image.
        Args:
            x: [B, N, patch_dim] where patch_dim = decoder_patch_size^2 * 3
        Returns:
            img: [B, 3, H, W]
        """
        return self.decoder.unpatchify(x, original_image_size=(self.img_size, self.img_size))

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patch targets for reconstruction loss.
        Args:
            imgs: [B, C, H, W]
        Returns:
            patches: [B, N, decoder_patch_size^2 * C]
        """
        p = self.decoder_patch_size
        B, C, H, W = imgs.shape
        if H % p != 0 or W % p != 0:
            raise ValueError(
                f"Input resolution ({H}, {W}) must be divisible by decoder patch_size {p}."
            )

        h = H // p
        w = W // p
        if h * w != self.num_patches:
            raise ValueError(
                f"Patchified token count {h * w} does not match model num_patches {self.num_patches}."
            )
        patches = imgs.reshape(B, C, h, p, w, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
        return patches
    
    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor,
                     z_proj: torch.Tensor, teacher_feat: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute losses:
        - loss_rec: Reconstruction loss (MSE on masked patches)
        - loss_align: Cosine similarity between student projector and teacher
        - loss_reg: L2 regularization on bottleneck
        """
        # 1. Reconstruction loss (MSE on masked patches only)
        # Target: patchify the original image
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs = F.interpolate(
                imgs,
                size=(self.img_size, self.img_size),
                mode='bicubic',
                align_corners=False,
            )
        target = self.patchify(imgs)  # [B, N, patch_size^2 * 3]
        
        # Normalize target (similar to MAE)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5
        
        # Compute loss only on masked patches
        rec_per_patch = (pred - target) ** 2
        rec_per_patch = rec_per_patch.mean(dim=-1)  # [B, N]
        mask_sum = mask.sum()
        if mask_sum.item() > 0:
            loss_rec = (rec_per_patch * mask).sum() / mask_sum  # mean over masked patches
        else:
            # mask_ratio=0 during online eval: fallback to full-patch reconstruction MSE.
            loss_rec = rec_per_patch.mean()
        
        # 2. Alignment loss (cosine similarity)
        # z_proj: [B, N_visible, teacher_dim]
        # teacher_feat: [B, teacher_dim]
        
        # Pool student features (mean over visible patches)
        z_pooled = z_proj.mean(dim=1)  # [B, teacher_dim]
        
        # Cosine similarity loss (1 - cos_sim)
        cos_sim = F.cosine_similarity(z_pooled, teacher_feat, dim=-1)
        loss_align = 1 - cos_sim.mean()
        
        # 3. Regularization loss (L2 on bottleneck)
        loss_reg = (z ** 2).mean()
        
        # Total loss
        loss_total = (self.loss_rec_weight * loss_rec + 
                     self.loss_align_weight * loss_align + 
                     self.loss_reg_weight * loss_reg)
        
        return {
            'loss': loss_total,
            'loss_rec': loss_rec,
            'loss_align': loss_align,
            'loss_reg': loss_reg,
        }
    
    def forward(self, imgs: torch.Tensor, mask_ratio: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass for training/inference.
        Args:
            imgs: [B, 3, H, W] input images
            mask_ratio: optional override for masking ratio
        Returns:
            reconstructed images: [B, 3, H, W]
        """
        # 1. Teacher forward (frozen)
        with torch.no_grad():
            teacher_feat = self.forward_teacher(imgs)
        
        # 2. Student forward (masked)
        z, z_proj, mask, ids_restore = self.forward_student(imgs, mask_ratio)
        
        # 3. Decoder forward
        pred = self.forward_decoder(z, ids_restore)  # [B, N, patch_dim]
        
        # 4. Compute losses and store for retrieval
        losses = self.forward_loss(imgs, pred, mask, z_proj, teacher_feat, z)
        self.last_losses = losses
        
        # 5. Decode to image
        pred_img = self.unpatchify(pred)
        # Normalize to [0, 1]
        pred_img = torch.clamp(pred_img, 0, 1)
        
        return pred_img
    
    def get_last_losses(self) -> Dict[str, torch.Tensor]:
        """Get the losses from the last forward pass"""
        return getattr(self, 'last_losses', {})
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to bottleneck latent (for inference)"""
        z, _, _, _ = self.forward_student(x, mask_ratio=0.0)  # No masking
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from bottleneck (for inference)"""
        B = z.size(0)
        N = self.num_patches
        
        # Create dummy ids_restore (assume all patches are visible, in order)
        ids_restore = torch.arange(N, device=z.device).unsqueeze(0).expand(B, -1)
        
        pred = self.forward_decoder(z, ids_restore)
        img = self.unpatchify(pred)
        return torch.clamp(img, 0, 1)
