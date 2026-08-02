from __future__ import annotations

import torch
import torch.nn as nn


class LightVJEPAEncoder(nn.Module):
    """Lightweight, JEPA-style ViT image encoder.
    
    """

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 96,
        depth: int = 4,
        num_heads: int = 3,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size ({image_size}) must be divisible by patch_size ({patch_size}).")
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.in_channels = in_channels
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, latent_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of RGB images into latent feature vectors.

        Args:
            images: Channels-last images ``(B, H, W, C)``, as produced by ``mdp.image``.

        Returns:
            Latent features of shape ``(B, latent_dim)``.
        """
        images = images.permute(0, 3, 1, 2)
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens.mean(dim=1))