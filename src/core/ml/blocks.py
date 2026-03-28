import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Applies feature-wise linear modulation: scales and shifts feature maps using illumination-derived gamma and beta."""

    def __init__(self, illum_dim, num_channels):
        """Initialise FiLM layer mapping illum_dim -> gamma and beta of size num_channels."""
        super().__init__()
        self.linear = nn.Linear(illum_dim, 2 * num_channels)

    def forward(self, x, illum_embedding):
        """Apply gamma * x + beta where gamma and beta are predicted from illum_embedding."""
        gb    = self.linear(illum_embedding)          # [B, 2*C]
        gamma, beta = gb.chunk(2, dim=1)              # [B, C], [B, C]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)     # [B, C, 1, 1]
        beta  = beta.unsqueeze(-1).unsqueeze(-1)      # [B, C, 1, 1]
        return gamma * x + beta


class ConvBlock(nn.Module):
    """Two conv-BN-ReLU layers with a FiLM modulation applied after the second activation."""

    def __init__(self, in_channels, out_channels, illum_dim):
        """Initialise conv block with two conv layers and a FiLM layer."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.film = FiLM(illum_dim, out_channels)

    def forward(self, x, illum_embedding):
        """Forward pass through conv layers then FiLM modulation."""
        x = self.block(x)
        return self.film(x, illum_embedding)


class IlluminationEncoder(nn.Module):
    """Small CNN that encodes an illumination quadrant into a flat embedding vector."""

    def __init__(self, illum_dim):
        """Initialise illumination encoder with three conv layers and a linear projection to illum_dim."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 32 -> 16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 16 -> 8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                  # 8 -> 1x1
            nn.Flatten(),                             # [B, 64]
        )
        self.proj = nn.Linear(64, illum_dim)

    def forward(self, illum_q):
        """Encode illumination quadrant [B, 1, H, W] to embedding [B, illum_dim]."""
        return self.proj(self.encoder(illum_q))


if __name__ == "__main__":
    illum_dim = 128
    B, C, H, W = 2, 32, 64, 64

    illum_q   = torch.randn(B, 1, 32, 32)
    x         = torch.randn(B, C, H, W)
    illum_emb = torch.randn(B, illum_dim)

    enc = IlluminationEncoder(illum_dim)
    emb = enc(illum_q)
    print(f"IlluminationEncoder: {tuple(illum_q.shape)} -> {tuple(emb.shape)}")

    film = FiLM(illum_dim, C)
    out  = film(x, illum_emb)
    print(f"FiLM:                {tuple(x.shape)} -> {tuple(out.shape)}")

    block = ConvBlock(1, C, illum_dim)
    out   = block(torch.randn(B, 1, H, W), illum_emb)
    print(f"ConvBlock:           {(B, 1, H, W)} -> {tuple(out.shape)}")

    print("\nAll checks passed.")