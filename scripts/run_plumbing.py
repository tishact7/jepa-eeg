import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.dataset import load_subject, patchify, fit_norm_stats, apply_norm
from src.models import Encoder, Predictor
from src.train_jepa import train_jepa
from src.evaluate import extract_features, linear_probe, random_encoder, tsne_plot
from src.utils import load_checkpoint, env_info, seed_everything, sample_masks, gather_patches, make_generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dir", default="results/plumbing")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {"args": vars(args), "env": env_info(), "checks": {}}

    X, y = load_subject(1)
    mean, std = fit_norm_stats(X)
    P = patchify(apply_norm(X, mean, std))
    ok1 = X.shape[1:] == (22, 500) and P.shape[1:] == (10, 22, 50) and bool(np.isfinite(P).all())

    seed_everything(args.seed)
    cidx, tidx = sample_masks(args.batch_size, make_generator(args.seed))
    b = torch.from_numpy(P[:args.batch_size]).to(device)
    cp = gather_patches(b, cidx.to(device))
    tp = gather_patches(b, tidx.to(device))
    enc_t, pred_t = Encoder().to(device), Predictor().to(device)
    pout = pred_t(enc_t(cp, cidx.to(device)), tidx.to(device))
    ok1 = ok1 and tuple(cp.shape[1:]) == (6, 22, 50) and tuple(tp.shape[1:]) == (2, 22, 50) and tuple(pout.shape[1:]) == (2, 128)
    summary["checks"]["1_shapes"] = {
        "pass": bool(ok1), "trials": list(X.shape), "patches": list(P.shape),
        "context": list(cp.shape), "target": list(tp.shape), "predictor_out": list(pout.shape)}
    print("check 1 shapes:", "PASS" if ok1 else "FAIL", list(X.shape), list(P.shape))

    out = train_jepa(P, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                     out_dir=args.out_dir, device=device)
    l20 = np.array(out["step_losses"][:20])
    strict = bool(np.all(np.diff(l20) < 0))
    ma = np.convolve(l20, np.ones(5) / 5, mode="valid")
    ma_dec = bool(ma[-1] < ma[0])
    last_std = out["history"][-1]["target_std"]
    summary["checks"]["2_loss"] = {
        "pass": ma_dec, "strict_monotonic_first20": strict,
        "decreasing_transitions": int((np.diff(l20) < 0).sum()), "of": 19,
        "ma5_first": float(ma[0]), "ma5_last": float(ma[-1]), "ma5_decreased": ma_dec,
        "final_target_std": last_std, "collapse": bool(last_std < 0.05)}
    print("check 2 loss: strict monotonic =", strict, "| MA decreased =", ma_dec,
          "| transitions", int((np.diff(l20) < 0).sum()), "of 19")

    ckpt_path = f"{args.out_dir}/ckpt_epoch1.pt"
    enc_a, pred_a = Encoder().to(device), Predictor().to(device)
    ep, ls, _ = load_checkpoint(ckpt_path, {"encoder": enc_a, "predictor": pred_a}, map_location=device)
    enc_b, pred_b = Encoder().to(device), Predictor().to(device)
    load_checkpoint(ckpt_path, {"encoder": enc_b, "predictor": pred_b}, map_location=device)
    enc_a.eval(); enc_b.eval()
    idx = torch.arange(10, device=device).expand(8, -1)
    fixed = torch.from_numpy(P[:8]).to(device)
    with torch.no_grad():
        same = bool(torch.equal(enc_a(fixed, idx), enc_b(fixed, idx)))
    state_ok = all(torch.equal(p, q) for p, q in zip(enc_a.state_dict().values(), enc_b.state_dict().values()))
    ok3 = same and state_ok and ep == 1
    summary["checks"]["3_checkpoint"] = {"pass": bool(ok3), "reloaded_epoch": ep,
                                         "outputs_match": same, "weights_match": state_ok,
                                         "epoch2_ckpt_exists": os.path.exists(f"{args.out_dir}/ckpt_epoch2.pt")}
    print("check 3 checkpoint:", "PASS" if ok3 else "FAIL", "| epoch", ep)

    tr, ev = train_test_split(np.arange(len(P)), test_size=0.2, stratify=y, random_state=0)
    enc = Encoder().to(device)
    load_checkpoint(f"{args.out_dir}/ckpt_epoch{args.epochs}.pt", {"encoder": enc}, map_location=device)
    ftr, fev = extract_features(enc, P[tr], device), extract_features(enc, P[ev], device)
    jr = linear_probe(ftr, y[tr], fev, y[ev], device=device)
    rnd = random_encoder(0, device)
    rr = linear_probe(extract_features(rnd, P[tr], device), y[tr],
                      extract_features(rnd, P[ev], device), y[ev], device=device)
    tsne_plot(fev, y[ev], f"{args.out_dir}/tsne_dummy.png", title="S1 dummy split, JEPA")
    ok4 = ftr.shape == (len(tr), 128) and fev.shape == (len(ev), 128) and 0.0 <= jr["eval_acc"] <= 1.0
    summary["checks"]["4_probe"] = {"pass": bool(ok4), "train_n": len(tr), "eval_n": len(ev),
                                    "jepa": jr, "random_encoder": rr,
                                    "note": "dummy within-subject split, not a result"}
    print("check 4 probe:", "PASS" if ok4 else "FAIL", "| jepa", jr, "| random", rr)

    summary["all_pass"] = all(c["pass"] for c in summary["checks"].values())
    with open(f"{args.out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("ALL CHECKS PASS" if summary["all_pass"] else "SOME CHECKS FAILED, see summary.json")


if __name__ == "__main__":
    main()
