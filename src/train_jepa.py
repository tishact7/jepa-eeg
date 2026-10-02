import copy
import torch
import torch.nn.functional as F
from src.models import Encoder, Predictor
from src.utils import (sample_masks, gather_patches, make_generator,
                       seed_everything, save_checkpoint, JsonlLogger)


@torch.no_grad()
def ema_update(online, target, momentum=0.996):
    for po, pt in zip(online.parameters(), target.parameters()):
        pt.mul_(momentum).add_(po.detach(), alpha=1 - momentum)


def build_jepa(device):
    enc, pred = Encoder().to(device), Predictor().to(device)
    tgt = copy.deepcopy(enc)
    for p in tgt.parameters():
        p.requires_grad = False  # stop-gradient: updated only by EMA
    tgt.eval()
    return enc, pred, tgt


def jepa_step(enc, pred, tgt, patches, optimizer, generator, momentum=0.996):
    """patches: (B,10,22,50) on device. Returns (loss, collapse_std)."""
    dev = patches.device
    ctx_idx, tgt_idx = sample_masks(patches.shape[0], generator)
    ctx_idx, tgt_idx = ctx_idx.to(dev), tgt_idx.to(dev)

    ctx = gather_patches(patches, ctx_idx)   # (B,6,22,50)
    tp = gather_patches(patches, tgt_idx)    # (B,2,22,50)

    pred_lat = pred(enc(ctx, ctx_idx), tgt_idx)   # (B,2,128)
    with torch.no_grad():
        tgt_lat = tgt(tp, tgt_idx)                # (B,2,128)

    loss = F.mse_loss(pred_lat, tgt_lat)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    ema_update(enc, tgt, momentum)

    # collapse monitor: mean per-dimension std of target latents (near 0 = collapse)
    collapse_std = tgt_lat.reshape(-1, tgt_lat.shape[-1]).std(0).mean().item()
    return loss.item(), collapse_std


def train_jepa(patches_np, epochs, batch_size, seed, out_dir, lr=5e-4,
               weight_decay=0.05, momentum=0.996, device="cuda", log_first=20):
    """Unlabeled JEPA pretraining. patches_np: (n,10,22,50) float32."""
    seed_everything(seed)
    mask_gen, shuffle_gen = make_generator(seed), make_generator(seed + 1)
    enc, pred, tgt = build_jepa(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=lr, weight_decay=weight_decay)
    logger = JsonlLogger(f"{out_dir}/train_log.jsonl")

    X = torch.from_numpy(patches_np).to(device)
    n = X.shape[0]
    step, step_losses, history = 0, [], []

    for epoch in range(1, epochs + 1):
        enc.train(); pred.train()
        order = torch.randperm(n, generator=shuffle_gen).to(device)
        ep_losses, ep_std = [], []
        for b in range(n // batch_size):  # drop last partial batch
            batch = X[order[b * batch_size:(b + 1) * batch_size]]
            loss, cstd = jepa_step(enc, pred, tgt, batch, opt, mask_gen, momentum)
            step += 1
            step_losses.append(loss); ep_losses.append(loss); ep_std.append(cstd)
            if step <= log_first:
                print(f"step {step:3d} | loss {loss:.5f} | target std {cstd:.4f}")
            logger.log(kind="step", epoch=epoch, step=step, loss=loss, target_std=cstd)
        ep_loss = sum(ep_losses) / len(ep_losses)
        ep_cstd = sum(ep_std) / len(ep_std)
        history.append({"epoch": epoch, "loss": ep_loss, "target_std": ep_cstd})
        logger.log(kind="epoch", epoch=epoch, loss=ep_loss, target_std=ep_cstd)
        print(f"epoch {epoch} | mean loss {ep_loss:.5f} | target std {ep_cstd:.4f}")
        save_checkpoint(f"{out_dir}/ckpt_epoch{epoch}.pt", epoch,
                        {"encoder": enc, "predictor": pred, "target": tgt}, opt, ep_loss)

    return {"encoder": enc, "predictor": pred, "target": tgt, "optimizer": opt,
            "step_losses": step_losses, "history": history}
