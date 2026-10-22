import math
import time
import torch
import torch.nn.functional as F
from src.models import BaselineClassifier, count_params, N_PATCHES
from src.utils import seed_everything, make_generator, JsonlLogger


def train_baseline(patches_np, labels_np, epochs, batch_size, seed, lr, out_dir,
                   device="cuda", weight_decay=0.05, save_epochs=(10, 20, 30, 40, 50)):
    seed_everything(seed)
    shuffle_gen = make_generator(seed + 1)
    model = BaselineClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    tag = f"lr{lr:g}"
    logger = JsonlLogger(f"{out_dir}/baseline_{tag}_log.jsonl")
    X = torch.from_numpy(patches_np).to(device)
    y = torch.from_numpy(labels_np).to(device)
    n = X.shape[0]
    epoch_losses, ckpt_paths, steps, finite = [], {}, 0, True
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n, generator=shuffle_gen).to(device)
        losses = []
        for b in range(n // batch_size):
            idx = order[b * batch_size:(b + 1) * batch_size]
            loss = F.cross_entropy(model(X[idx]), y[idx])
            if not math.isfinite(loss.item()):
                finite = False
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
            losses.append(loss.item())
        if not finite:
            logger.log(kind="diverged", epoch=epoch, lr=lr)
            break
        ep_loss = sum(losses) / len(losses)
        epoch_losses.append(ep_loss)
        logger.log(kind="epoch", epoch=epoch, loss=ep_loss, lr=lr)
        if epoch in save_epochs:
            path = f"{out_dir}/baseline_{tag}_epoch{epoch}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict()}, path)
            ckpt_paths[epoch] = path
    return {"epoch_losses": epoch_losses, "steps": steps, "finite": finite,
            "params": count_params(model), "wall_seconds": time.time() - start,
            "ckpt_paths": ckpt_paths}


@torch.no_grad()
def baseline_counts(ckpt_path, X, labels, device, batch_size=128):
    model = BaselineClassifier().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()
    correct = 0
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        logits = model(xb)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("non-finite logits")
        yb = torch.as_tensor(labels[i:i + batch_size]).to(device)
        correct += int((logits.argmax(1) == yb).sum().item())
    return correct, int(len(labels))
