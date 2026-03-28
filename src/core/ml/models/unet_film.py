# src/core/ml/models/unet_film.py
import torch
import torch.nn as nn
from src.core.ml.registry import register_model
from src.core.ml.blocks import ConvBlock, IlluminationEncoder


@register_model("unet_film")
class UNetFiLM(nn.Module):
    """UNet with FiLM conditioning from an illumination encoder, predicting wafer intensity and resist profile."""

    def __init__(self, config):
        """Initialise UNetFiLM from config with keys: channels (default 32), illum_dim (default 128)."""
        super().__init__()
        model_cfg = config["model"]
        ch        = model_cfg.get("channels", 32)
        illum_dim = model_cfg.get("illum_dim", 128)

        self.illum_encoder = IlluminationEncoder(illum_dim)

        # Encoder
        self.enc1 = ConvBlock(1,      ch,     illum_dim)
        self.enc2 = ConvBlock(ch,     ch * 2, illum_dim)
        self.enc3 = ConvBlock(ch * 2, ch * 4, illum_dim)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(ch * 4, ch * 8, illum_dim)

        # Decoder
        self.up3  = nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(ch * 8, ch * 4, illum_dim)

        self.up2  = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(ch * 4, ch * 2, illum_dim)

        self.up1  = nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(ch * 2, ch, illum_dim)

        # Output heads
        self.intensity_head = nn.Conv2d(ch, 1, kernel_size=1)
        self.resist_head    = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, mask, illum_q):
        """Forward pass returning (intensity, resist), each [B, 1, H, W] with values in [0, 1]."""
        illum_emb = self.illum_encoder(illum_q)

        # Encoder
        e1 = self.enc1(mask,          illum_emb)
        e2 = self.enc2(self.pool(e1), illum_emb)
        e3 = self.enc3(self.pool(e2), illum_emb)

        # Bottleneck
        b  = self.bottleneck(self.pool(e3), illum_emb)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1), illum_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), illum_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), illum_emb)

        intensity = torch.sigmoid(self.intensity_head(d1))
        resist    = torch.sigmoid(self.resist_head(d1))

        return intensity, resist


if __name__ == "__main__":
    config = {
        "model": {
            "name":      "unet_film",
            "channels":  16,
            "illum_dim": 128,
        }
    }

    model     = UNetFiLM(config)
    mask      = torch.randn(2, 1, 128, 128)
    illum_q   = torch.randn(2, 1, 32, 32)

    intensity, resist = model(mask, illum_q)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total_params:,}")
    print(f"Input  mask      : {tuple(mask.shape)}")
    print(f"Input  illum_q   : {tuple(illum_q.shape)}")
    print(f"Output intensity : {tuple(intensity.shape)}  min={intensity.min():.3f}  max={intensity.max():.3f}")
    print(f"Output resist    : {tuple(resist.shape)}     min={resist.min():.3f}  max={resist.max():.3f}")

    print("\nAll checks passed.")