#
# conditional DiT video model
# uses adaLN for the generation conditioning
# 
# 

import torch
from model.encdec import Encoder, Decoder
from torch import nn
from dataclasses import dataclass
from einops import rearrange
from torch.nn import functional as F
from model.rotary_embedding import RotaryEmbedding3D, apply_rotary_3d
from model.rotary_embeddings import RotaryEmbedding, apply_rotary_emb

@dataclass
class DiffusionForcingDiTConfig:
    image_channels: int
    patch_size: int
    embed_dim: int
    num_patch_rows: int
    num_patch_cols: int
    num_classes: int
    num_conditions: int
    # transformer params
    num_heads: int
    num_layers: int


class AdaLayerNorm(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 3 * embed_dim),
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.ln = nn.LayerNorm(embed_dim, elementwise_affine=False)

    def forward(self, x, cond):
        h = self.ln(x)
        cond = self.mlp(cond)

        gamma, beta, alpha = cond.chunk(3, dim=-1)

        h = (1 + gamma) * h + beta

        return h, (1 + alpha)

# class AttentionWith3DRope(nn.Module):
#     def __init__(self, embed_dim: int, num_heads: int):
#         super().__init__()
#         self.rope = RotaryEmbedding3D(embed_dim // num_heads)
#         self.nH = num_heads
#         self.D = embed_dim
#         self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
#         self.c_proj = nn.Linear(embed_dim, embed_dim)

#     # x: (B, S, D)
#     def forward(self, x, frame_height, frame_width, attn_mask):
#         qkv = self.qkv(x)
#         # (B, S, D)
#         q, k, v = qkv.chunk(3, dim=-1)
#         s = q.shape[1]
#         t, h, w = s // (frame_height * frame_width), frame_height, frame_width

#         rope = self.rope(t, frame_height, frame_width)

#         # reshape for rotary
#         q = rearrange(q, 'b (t h w) (nh d) -> b t h w nh d', nh=self.nH, h=frame_height, w=frame_width)
#         k = rearrange(k, 'b (t h w) (nh d) -> b t h w nh d', nh=self.nH, h=frame_height, w=frame_width)
#         v = rearrange(v, 'b (t h w) (nh d) -> b t h w nh d', nh=self.nH, h=frame_height, w=frame_width)

#         q = apply_rotary_3d(q, *rope)
#         k = apply_rotary_3d(k, *rope)
#         v = apply_rotary_3d(v, *rope)

#         # 24, 20, 20

#         q = rearrange(q, 'b t h w nh d -> b nh (t h w) d')
#         k = rearrange(k, 'b t h w nh d -> b nh (t h w) d')
#         v = rearrange(v, 'b t h w nh d -> b nh (t h w) d')

#         out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
#         out = rearrange(out, 'b nh s d -> b s (nh d)')
#         out = self.c_proj(out)
#         return out

# class RotaryAttention(nn.Module):
#     def __init__(self, embed_dim: int, num_heads: int):
#         super().__init__()

#     def forward(self, x, frame_height, frame_width):
#         return self.attn(x, frame_height, frame_width, self.compute_attention_mask(frame_height, frame_width))


# Attention across the image
class SpaceAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        # we split the rope into two, one for each axis.
        self.rope = RotaryEmbedding(embed_dim // num_heads // 2, freqs_for="pixel")
        self.nH = num_heads

    # x: (B, T, H, W, D)
    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        _, _, h, w, _ = x.shape
        # (B, T, H, W, D)
        q = rearrange(q, 'b t h w (nh d) -> b t nh h w d', nh=self.nH)
        k = rearrange(k, 'b t h w (nh d) -> b t nh h w d', nh=self.nH) 
        v = rearrange(v, 'b t h w (nh d) -> b t nh h w d', nh=self.nH)

        # this should get us (H, W, D) shaped frequency vector
        freqs = self.rope.get_axial_freqs(h, w)

        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
        v = apply_rotary_emb(freqs, v)

        # (B, T, H, W, num_heads, head_dim)
        # flatten the hw before doing sdpa
        q = rearrange(q, 'b t nh h w d -> b t (h w) nh d')
        k = rearrange(k, 'b t nh h w d -> b t (h w) nh d')
        v = rearrange(v, 'b t nh h w d -> b t (h w) nh d')

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
        out = rearrange(out, 'b t (h w) nh d -> b t h w (nh d)', h=h, w=w)

        out = self.c_proj(out)
        return out

class TimeAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        # we use language-style rotary embeddings for the time axis (works for autoregressive)
        self.rope = RotaryEmbedding(embed_dim // num_heads, freqs_for="lang")
        self.nH = num_heads
    
    # 
    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, 'b t h w (nh d) -> b h w t nh d', nh=self.nH)
        k = rearrange(k, 'b t h w (nh d) -> b h w t nh d', nh=self.nH)
        v = rearrange(v, 'b t h w (nh d) -> b h w t nh d', nh=self.nH)

        q = self.rope.rotate_queries_or_keys(q)
        k = self.rope.rotate_queries_or_keys(k)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
        out = rearrange(out, 'b h w t nh d -> b t h w (nh d)')

        out = self.c_proj(out)

        return out


# for this, we will use Llama style attention blocks.
class SpaceTimeTransformerBlock(nn.Module):
    def __init__(self, 
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()

        self.space_attn_norm = AdaLayerNorm(embed_dim)
        self.time_attn_norm = AdaLayerNorm(embed_dim)
        self.space_mlp_norm = AdaLayerNorm(embed_dim)
        self.time_mlp_norm = AdaLayerNorm(embed_dim)

        self.space_mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.space_attn = SpaceAttention(embed_dim, num_heads)
        self.time_attn = TimeAttention(embed_dim, num_heads)

    def forward(self, x, t):
        # 1. space attention
        h, alpha = self.space_attn_norm(x, t)                                   # (B, S, D)
        x = x + self.space_attn(h) * alpha
        # 2. space mlp
        h, alpha = self.space_mlp_norm(x, t)    
        x = x + self.space_mlp(h) * alpha
        # 3. time attention
        h, alpha = self.time_attn_norm(x, t)
        x = x + self.time_attn(h) * alpha
        # 4. time mlp
        h, alpha = self.time_mlp_norm(x, t)
        x = x + self.time_mlp(h) * alpha
        return x

# a video DiT model
class DiffusionForcingDiT(nn.Module):
    def __init__(self, 
        config: DiffusionForcingDiTConfig,
    ):
        super().__init__()

        self.encoder = Encoder(
            patch_size=config.patch_size,
            image_channels=config.image_channels,
            embed_dim=config.embed_dim,
            num_patch_rows=config.num_patch_rows,
            num_patch_cols=config.num_patch_cols,
        )

        self.c_emb = nn.Embedding(config.num_classes, config.embed_dim)
        self.cond_emb = nn.Sequential(
            nn.Linear(1 + config.embed_dim * config.num_conditions, config.embed_dim * 2),
            nn.GELU(),
            nn.Linear(config.embed_dim * 2, config.embed_dim),
        )

        self.decoder = Decoder(
            patch_size=config.patch_size,
            image_channels=config.image_channels,
            embed_dim=config.embed_dim,
            num_patch_rows=config.num_patch_rows,
            num_patch_cols=config.num_patch_cols,
        )

        self.blocks = nn.ModuleList([
            SpaceTimeTransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
            )
            for _ in range(config.num_layers)
        ])

    def from_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
    
    def forward(self, x, t, c):
        # x: (B, T, C, H, W)
        x = self.encoder(x)
        # x: (B, T, H, W, D)
        c_vec = self.c_emb(c) # (B, num_conditions, D)

        if c_vec.ndim == 3:
            c_vec = rearrange(c_vec, "b n d -> b (n d)")

        cond_vec = self.cond_emb(torch.cat([t.unsqueeze(-1), c_vec], dim=-1))
        cond = rearrange(cond_vec, "b d -> b 1 1 1 d")

        for block in self.blocks:
            x = block(x, cond)

        x = self.decoder(x)

        return x
    
def test_dit():
    mnist_config = DiffusionForcingDiTConfig(
        patch_size=4,
        num_layers=6,
        num_heads=4,
        embed_dim=256,
        image_channels=3,
        num_patch_rows=16,
        num_patch_cols=16,
        num_classes=10,
    )
    dit = DiffusionForcingDiT(mnist_config)
    dit.to("cuda")

    # x: (B, T, C, H, W)
    x = torch.randn(1, 4, 3, 64, 64).to("cuda")
    t = torch.tensor([0]).to("cuda")
    c = torch.tensor([0]).to("cuda")

    y = dit(x, t, c)

    assert x.shape == y.shape

if __name__ == "__main__":
    test_dit()