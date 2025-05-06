# a diffusion transformer model written from scratch.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import einops

# you should encode more complex frames.
class Encoder(nn.Module):
    def __init__(
        self, 
        patch_size: int,
        image_channels: int,
        embed_dim: int,
        num_patch_rows: int,
        num_patch_cols: int,
    ):
        super().__init__()
        # our patches should be 16x16
        self.patch_size = patch_size
        self.encoder = nn.Conv2d(
            image_channels, 
            embed_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size,
        )
        nn.init.trunc_normal_(self.encoder.weight, std=0.02)

    # INPUT
    # x: [B, T, C, H, W]
    # returns: [B, T, H, W, D]
    def forward(self, x):
        b, t, _, _, _ = x.shape
        x = einops.rearrange(x, "b t c h w -> (b t) c h w")
        # (B x T, C, H, W) -> (B x T, D, H, W)
        h = self.encoder(x)
        # (B x T, D, H, W) -> (B, T, D, H, W)
        h = einops.rearrange(h, "(b t) d h w -> b t h w d", b=b, t=t)

        return h

# a 3d positional encoding scheme
class VideoRoPE(nn.Module):
    # 
    def __init__(self, 
        embed_dim: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.r_dim = embed_dim // 3
        self.c_dim = embed_dim // 3
        self.t_dim = embed_dim - self.x_dim - self.y_dim

    # turn the x into a 3d positional encoding
    def forward(self, x, r, c, t):
        
        # x: [B, H, W, D]
        # r: [B, H, W]
        # c: [B, H, W]
        # t: [B, H, W]
        # return [B, H, W, D]
        pass


# decode the tokens
class Decoder(nn.Module):
    def __init__(self, 
        patch_size: int,
        image_channels: int,
        embed_dim: int,
        num_patch_rows: int,
        num_patch_cols: int,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.decoder = nn.ConvTranspose2d(
            embed_dim, 
            image_channels, 
            kernel_size=self.patch_size, 
            stride=self.patch_size,
        )
        nn.init.trunc_normal_(self.decoder.weight, std=0.02)
        self.num_patch_rows = num_patch_rows
        self.num_patch_cols = num_patch_cols

    # INPUT
    # x: [B, embed_dim, H_p, W_p]
    def forward(self, x):
        # b t h w d -> b t d h w
        b, t, _, _, _ = x.shape
        x = einops.rearrange(x, "b t h w d -> (b t) d h w")
        # b t d h w -> b t c h w
        return einops.rearrange(self.decoder(x), "(b t) c h w -> b t c h w", b=b, t=t)

# small test
if __name__ == "__main__":
    random_image = torch.randn(1, 3, 64, 64)
    encoder = Encoder(
        patch_size=4, 
        image_channels=3, 
        embed_dim=128, 
        num_patch_rows=16, 
        num_patch_cols=16
    )
    h = encoder(random_image)
