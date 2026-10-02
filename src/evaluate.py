import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.models import Encoder, N_PATCHES
from src.utils import seed_everything, make_generator


@torch.no_grad()
def extract_features(encoder, X, device, batch_size=128):
    encoder.eval()
    feats = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        idx = torch.arange(N_PATCHES, device=device).expand(xb.shape[0], -1)
        feats.append(encoder(xb, idx).mean(1).cpu())
    return torch.cat(feats)


def linear_probe(train_feats, train_labels, eval_feats, eval_labels,
                 epochs=100, lr=1e-3, batch_size=64, seed=0, device="cpu"):
    seed_everything(seed)
    gen = make_generator(seed)
    mu = train_feats.mean(0, keepdim=True)
    sd = train_feats.std(0, keepdim=True) + 1e-8
    xtr = ((train_feats - mu) / sd).to(device)
    xev = ((eval_feats - mu) / sd).to(device)
    ytr = torch.as_tensor(train_labels).to(device)
    yev = torch.as_tensor(eval_labels).to(device)

    probe = nn.Linear(xtr.shape[1], 4).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    n = xtr.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, batch_size):
            b = order[i:i + batch_size]
            loss = F.cross_entropy(probe(xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        train_acc = (probe(xtr).argmax(1) == ytr).float().mean().item()
        eval_acc = (probe(xev).argmax(1) == yev).float().mean().item()
    return {"train_acc": train_acc, "eval_acc": eval_acc}


def random_encoder(seed, device):
    seed_everything(seed)
    return Encoder().to(device)


def tsne_plot(feats, labels, save_path, title="", seed=42, perplexity=30):
    emb = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
               init="pca").fit_transform(feats.numpy())
    names = ["left_hand", "right_hand", "feet", "tongue"]
    plt.figure(figsize=(7, 6))
    for c in range(4):
        m = np.asarray(labels) == c
        plt.scatter(emb[m, 0], emb[m, 1], s=14, alpha=0.7, label=names[c])
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
