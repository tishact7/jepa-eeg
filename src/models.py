import torch
import torch.nn as nn

N_PATCHES = 10
PATCH_DIM = 22 * 50  # 1100
LATENT_DIM = 128


class Encoder(nn.Module):
    """Patch embedding + learned positions + small transformer.
    Input: patches (B, n, 22, 50), idx (B, n) long = original patch positions.
    Output: latents (B, n, 128)."""

    def __init__(self, patch_dim=PATCH_DIM, dim=LATENT_DIM, depth=2, heads=4):
        super().__init__()
        self.embed = nn.Linear(patch_dim, dim)
        self.pos = nn.Embedding(N_PATCHES, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, patches, idx):
        B, n = patches.shape[:2]
        x = self.embed(patches.reshape(B, n, -1)) + self.pos(idx)
        return self.norm(self.tf(x))


class Predictor(nn.Module):
    """Takes context latents (B, nc, 128) and target positions (B, nt).
    Appends a learned mask token + target position embedding per target,
    returns predicted target latents (B, nt, 128)."""

    def __init__(self, dim=LATENT_DIM, depth=2, heads=4):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.pos = nn.Embedding(N_PATCHES, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ctx_latents, target_idx):
        B, nt = target_idx.shape
        q = self.mask_token.expand(B, nt, -1) + self.pos(target_idx)
        x = torch.cat([ctx_latents, q], dim=1)
        x = self.norm(self.tf(x))
        return x[:, -nt:]


class BaselineClassifier(nn.Module):
    """Same Encoder class on all 10 patches, mean-pool, Linear(128, 4)."""

    def __init__(self, n_classes=4):
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Linear(LATENT_DIM, n_classes)

    def forward(self, patches):
        B = patches.shape[0]
        idx = torch.arange(N_PATCHES, device=patches.device).expand(B, -1)
        return self.head(self.encoder(patches, idx).mean(1))


def count_params(m):
    return sum(p.numel() for p in m.parameters())
