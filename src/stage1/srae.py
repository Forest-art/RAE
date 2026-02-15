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
from typing import Optional, Tuple, Dict
from transformers import Dinov2Model, AutoConfig
from .decoders import GeneralDecoder
from .decoders.utils import ViTMAEConfig
import random


class PatchEmbed(nn.Module):
    """Image to Patch Embedding (from timm)"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class SRAE(nn.Module):
    """
    Semantic-Guided Regularized Autoencoder (S-RAE)
    
    Combines DINOv2 semantic understanding with MAE-style generation
    through a continuous latent space with semantic alignment.
    """
    
    def __init__(
        self,
        # Teacher/Student backbone config
        dinov2_model_name: str = "facebook/dinov2-base",
        img_size: int = 224,
        patch_size: int = 14,
        
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
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        self.loss_rec_weight = loss_rec_weight
        self.loss_align_weight = loss_align_weight
        self.loss_reg_weight = loss_reg_weight
        self.teacher_output_type = teacher_output_type
        
        # ========== Teacher (Frozen DINOv2) ==========
        print(f"Loading Teacher DINOv2 from {dinov2_model_name}...")
        self.teacher = Dinov2Model.from_pretrained(dinov2_model_name)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        teacher_dim = self.teacher.config.hidden_size
        print(f"Teacher loaded. Hidden dim: {teacher_dim}")
        
        # ========== Student (Trainable DINOv2) ==========
        print(f"Loading Student DINOv2 from {dinov2_model_name}...")
        self.student_encoder = Dinov2Model.from_pretrained(dinov2_model_name)
        student_dim = self.student_encoder.config.hidden_size
        print(f"Student loaded. Hidden dim: {student_dim}")
        
        # Remove student's head (we'll use our own)
        if hasattr(self.student_encoder, 'head'):
            self.student_encoder.head = nn.Identity()
        
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
            image_size=img_size,
            patch_size=patch_size,
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
        
        # Positional embedding for decoder (full set of patches)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_dim)  # +1 for CLS
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        
        print(f"S-RAE initialized:")
        print(f"  - Bottleneck dim: {bottleneck_dim}")
        print(f"  - Decoder dim: {decoder_dim}")
        print(f"  - Mask ratio: {mask_ratio}")
        
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
        # Resize to teacher's expected size if needed
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bicubic', align_corners=False)
        
        outputs = self.teacher(pixel_values=x, output_hidden_states=True)
        
        if self.teacher_output_type == "cls":
            # Use CLS token
            return outputs.last_hidden_state[:, 0]  # [B, D]
        else:
            # Use mean of patch tokens
            return outputs.last_hidden_state[:, 1:].mean(dim=1)  # [B, D]
    
    def forward_student(self, x: torch.Tensor, mask_ratio: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward through student encoder (with masking)
        Returns:
            z: bottleneck latent [B, N_visible, bottleneck_dim]
            z_proj: projected latent for alignment [B, N_visible, teacher_dim]
            mask: [B, N]
            ids_restore: [B, N]
        """
        mask_ratio = mask_ratio or self.mask_ratio
        
        # Resize if needed
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bicubic', align_corners=False)
        
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
        num_patches = self.student_encoder.embeddings.patch_embeddings.num_patches
        pos_embed = self.student_encoder.embeddings.position_embeddings  # [1, N+1, D]
        
        # Interpolate position embeddings if needed
        if pos_embed.shape[1] - 1 != num_patches:
            pos_embed = self.student_encoder.embeddings.interpolate_pos_encoding(
                pos_embed, self.img_size, self.img_size
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
        x_full = torch.cat([cls_token, x_visible], dim=1)  # [B, 1+N_visible, D]
        
        # Pass through transformer
        hidden_states = self.student_encoder.encoder(x_full).last_hidden_state
        
        # Extract patch tokens (exclude CLS)
        patch_tokens = hidden_states[:, 1:]  # [B, N_visible, D]
        
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
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_shuffle.unsqueeze(-1).expand(-1, -1, x_full.size(-1)))
        
        # Add positional embeddings
        x_full = x_full + self.decoder_pos_embed[:, 1:, :]  # Exclude CLS position
        
        # Add CLS token
        cls_token = self.mask_token.expand(B, 1, -1) + self.decoder_pos_embed[:, :1, :]
        x_full = torch.cat([cls_token, x_full], dim=1)  # [B, N+1, decoder_dim]
        
        # Pass through decoder
        outputs = self.decoder(x_full)
        logits = outputs.logits  # [B, N, patch_dim]
        
        return logits
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patch logits to image.
        Args:
            x: [B, N, patch_dim] where patch_dim = patch_size^2 * 3
        Returns:
            img: [B, 3, H, W]
        """
        return self.decoder.unpatchify(x)
    
    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor,
                     z_proj: torch.Tensor, teacher_feat: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute losses:
        - loss_rec: Reconstruction loss (MSE on masked patches)
        - loss_align: Cosine similarity between student projector and teacher
        - loss_reg: L2 regularization on bottleneck
        """
        B = imgs.size(0)
        
        # 1. Reconstruction loss (MSE on masked patches only)
        # Target: patchify the original image
        target = self.student_encoder.embeddings.patch_embeddings.projection(imgs)
        target = target.flatten(2).transpose(1, 2)  # [B, N, patch_dim]
        
        # Normalize target (similar to MAE)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5
        
        # Compute loss only on masked patches
        loss_rec = (pred - target) ** 2
        loss_rec = loss_rec.mean(dim=-1)  # [B, N]
        loss_rec = (loss_rec * mask).sum() / mask.sum()  # Mean over masked patches
        
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
