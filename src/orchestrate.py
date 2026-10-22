import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from fractions import Fraction

import torch

from src.evaluate import (extract_features, fit_probe, probe_counts,
                          random_encoder, tsne_plot)
from src.models import Encoder, count_params
from src.selection import (BASELINE_RATES, CHECKPOINTS, PROBE_EPOCHS, PROBE_RATES,
                           SEEDS, Accuracy, Candidate, candidate_grid, falsifier,
                           loss_trend_flag, select_on_s8)
from src.train_baseline import baseline_counts, train_baseline
from src.train_jepa import train_jepa


@dataclass(frozen=True)
class Plan:
    epochs: int = 50
    checkpoints: tuple = CHECKPOINTS
    probe_rates: tuple = PROBE_RATES
    probe_epochs: tuple = PROBE_EPOCHS
    baseline_rates: tuple = BASELINE_RATES
    batch_size: int = 32


FROZEN_PLAN = Plan()


def plan_json(plan):
    return {"epochs": plan.epochs, "checkpoints": list(plan.checkpoints),
            "probe_rates": [str(x) for x in plan.probe_rates],
            "probe_epochs": list(plan.probe_epochs),
            "baseline_rates": [str(x) for x in plan.baseline_rates],
            "batch_size": plan.batch_size}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"artifact changed after selection was recorded: {path}")


def _grid(model, seed, plan):
    return candidate_grid(model, seed, plan.checkpoints, plan.probe_rates,
                          plan.probe_epochs, plan.baseline_rates)


def _cand_json(c):
    return {"model": c.model, "seed": c.seed, "checkpoint_epoch": c.checkpoint_epoch,
            "learning_rate": str(c.learning_rate), "probe_epochs": c.probe_epochs,
            "s8_correct": c.s8.correct, "s8_total": c.s8.total}


def _load_encoder(path, device):
    enc = Encoder().to(device)
    enc.load_state_dict(torch.load(path, map_location=device)["models"]["encoder"])
    return enc


def _score_probes(model, seed, enc, Xtr, ytr, Xval, yval, ckpt_epoch, plan, device,
                  cands, probes, failures):
    ftr = extract_features(enc, Xtr, device)
    fval = extract_features(enc, Xval, device)
    if not (bool(torch.isfinite(ftr).all()) and bool(torch.isfinite(fval).all())):
        failures.append({"model": model, "checkpoint_epoch": ckpt_epoch,
                         "reason": "non-finite features"})
        return
    for lr in plan.probe_rates:
        for pe in plan.probe_epochs:
            pr = fit_probe(ftr, ytr, pe, float(lr), seed, device=device)
            if not pr["finite"]:
                failures.append({"model": model, "checkpoint_epoch": ckpt_epoch,
                                 "learning_rate": str(lr), "probe_epochs": pe,
                                 "reason": "non-finite probe"})
                continue
            try:
                correct, total = probe_counts(pr, fval, yval)
            except ValueError:
                failures.append({"model": model, "checkpoint_epoch": ckpt_epoch,
                                 "learning_rate": str(lr), "probe_epochs": pe,
                                 "reason": "non-finite S8 scoring"})
                continue
            cand = Candidate(model, seed, ckpt_epoch, lr, pe, Accuracy(correct, total))
            cands[model].append(cand)
            probes[(model, cand.key)] = pr


def run_seed(seed, data, out_dir, device, plan=FROZEN_PLAN, make_tsne=True):
    Xtr, ytr, Xval, yval = data["Xtr"], data["ytr"], data["Xval"], data["yval"]
    sdir = f"{out_dir}/seed{seed}"
    os.makedirs(sdir, exist_ok=True)
    record = {"seed": seed, "plan": plan_json(plan), "status": None,
              "failures": [], "resources": {}, "s9_accessed": False}
    failures = record["failures"]
    cands = {"jepa": [], "baseline": [], "random": []}
    probes = {}

    jdir = f"{sdir}/jepa"
    start = time.time()
    out = train_jepa(Xtr, plan.epochs, plan.batch_size, seed, jdir, device=device,
                     save_epochs=plan.checkpoints)
    jlosses = [h["loss"] for h in out["history"]]
    record["resources"]["jepa"] = {
        "trainable_params": count_params(out["encoder"]) + count_params(out["predictor"]),
        "encoder_params": count_params(out["encoder"]),
        "steps": len(out["step_losses"]), "epochs": plan.epochs,
        "wall_seconds": time.time() - start,
        "late_loss_trend_flag": loss_trend_flag(jlosses) if len(jlosses) == 50 else None}
    jepa_finite = all(math.isfinite(x) for x in out["step_losses"])
    del out
    if not jepa_finite:
        failures.append({"model": "jepa", "reason": "non-finite training loss"})

    if jepa_finite:
        if make_tsne:
            last = max(plan.checkpoints)
            enc = _load_encoder(f"{jdir}/ckpt_epoch{last}.pt", device)
            fval = extract_features(enc, Xval, device)
            if bool(torch.isfinite(fval).all()):
                tsne_plot(fval, yval, f"{sdir}/tsne_val_epoch{last}.png",
                          title=f"S8 t-SNE, JEPA epoch {last}, seed {seed}")
        for e in plan.checkpoints:
            enc = _load_encoder(f"{jdir}/ckpt_epoch{e}.pt", device)
            _score_probes("jepa", seed, enc, Xtr, ytr, Xval, yval, e, plan, device,
                          cands, probes, failures)

    renc = random_encoder(seed, device)
    rpath = f"{sdir}/random_encoder.pt"
    torch.save({"models": {"encoder": renc.state_dict()}}, rpath)
    _score_probes("random", seed, renc, Xtr, ytr, Xval, yval, 0, plan, device,
                  cands, probes, failures)

    bdir = f"{sdir}/baseline"
    os.makedirs(bdir, exist_ok=True)
    bpaths = {}
    record["resources"]["baseline"] = []
    for lr in plan.baseline_rates:
        tb = train_baseline(Xtr, ytr, plan.epochs, plan.batch_size, seed, float(lr), bdir,
                            device=device, save_epochs=plan.checkpoints)
        record["resources"]["baseline"].append({
            "learning_rate": str(lr), "trainable_params": tb["params"],
            "steps": tb["steps"], "epochs": plan.epochs,
            "wall_seconds": tb["wall_seconds"],
            "late_loss_trend_flag": loss_trend_flag(tb["epoch_losses"])
            if len(tb["epoch_losses"]) == 50 else None})
        if not tb["finite"]:
            failures.append({"model": "baseline", "learning_rate": str(lr),
                             "reason": "non-finite training loss"})
            continue
        for e in plan.checkpoints:
            try:
                correct, total = baseline_counts(tb["ckpt_paths"][e], Xval, yval, device)
            except (ValueError, KeyError):
                failures.append({"model": "baseline", "learning_rate": str(lr),
                                 "checkpoint_epoch": e, "reason": "non-finite S8 scoring"})
                continue
            cand = Candidate("baseline", seed, e, lr, 0, Accuracy(correct, total))
            cands["baseline"].append(cand)
            bpaths[cand.key] = tb["ckpt_paths"][e]

    record["candidates"] = [_cand_json(c) for m in cands for c in cands[m]]
    record["s8_evaluations"] = len(record["candidates"])

    selected = {}
    if not failures:
        try:
            for m in ("jepa", "baseline", "random"):
                selected[m] = select_on_s8(m, seed, cands[m], grid=_grid(m, seed, plan))
        except ValueError as exc:
            failures.append({"reason": str(exc)})

    if failures:
        record["status"] = "NOT_EVALUABLE"
    else:
        seldir = f"{sdir}/selected"
        os.makedirs(seldir, exist_ok=True)
        jc, bc, rc = selected["jepa"], selected["baseline"], selected["random"]
        jck = f"{jdir}/ckpt_epoch{jc.checkpoint_epoch}.pt"
        jprobe = f"{seldir}/jepa_probe.pt"
        torch.save(probes[("jepa", jc.key)], jprobe)
        rprobe = f"{seldir}/random_probe.pt"
        torch.save(probes[("random", rc.key)], rprobe)
        bck = bpaths[bc.key]
        record["selected"] = {
            "jepa": {**_cand_json(jc), "checkpoint": jck, "checkpoint_sha256": sha256(jck),
                     "probe": jprobe, "probe_sha256": sha256(jprobe)},
            "random": {**_cand_json(rc), "encoder": rpath, "encoder_sha256": sha256(rpath),
                       "probe": rprobe, "probe_sha256": sha256(rprobe)},
            "baseline": {**_cand_json(bc), "checkpoint": bck,
                         "checkpoint_sha256": sha256(bck)}}
        record["status"] = "SELECTED"

    record["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(f"{out_dir}/selection_record_seed{seed}.json", "w") as f:
        json.dump(record, f, indent=2, allow_nan=False)
    return record


def final_counts(record, Xtest, ytest, device):
    if record.get("status") != "SELECTED":
        raise ValueError("selection record is not SELECTED")
    sel = record["selected"]
    j, r, b = sel["jepa"], sel["random"], sel["baseline"]
    for path, digest in ((j["checkpoint"], j["checkpoint_sha256"]),
                         (j["probe"], j["probe_sha256"]),
                         (r["encoder"], r["encoder_sha256"]),
                         (r["probe"], r["probe_sha256"]),
                         (b["checkpoint"], b["checkpoint_sha256"])):
        verify_file(path, digest)
    jfeat = extract_features(_load_encoder(j["checkpoint"], device), Xtest, device)
    rfeat = extract_features(_load_encoder(r["encoder"], device), Xtest, device)
    jc, jt = probe_counts(torch.load(j["probe"], map_location="cpu"), jfeat, ytest)
    rc, rt = probe_counts(torch.load(r["probe"], map_location="cpu"), rfeat, ytest)
    bc, bt = baseline_counts(b["checkpoint"], Xtest, ytest, device)
    return {"jepa": Accuracy(jc, jt), "random": Accuracy(rc, rt), "baseline": Accuracy(bc, bt)}


def build_final_report(records, counts_by_seed):
    jepa = {s: c["jepa"] for s, c in counts_by_seed.items()}
    baseline = {s: c["baseline"] for s, c in counts_by_seed.items()}
    report = {"falsifier": falsifier(jepa, baseline),
              "per_seed": {str(s): {m: {"correct": a.correct, "total": a.total}
                                    for m, a in c.items()}
                           for s, c in counts_by_seed.items()},
              "selected": {str(r["seed"]): {m: {k: v for k, v in d.items()
                                                if k in ("checkpoint_epoch", "learning_rate",
                                                         "probe_epochs", "s8_correct", "s8_total")}
                                            for m, d in r["selected"].items()}
                           for r in records},
              "resources": {str(r["seed"]): r["resources"] for r in records},
              "s8_evaluations": {str(r["seed"]): r["s8_evaluations"] for r in records}}
    if report["falsifier"]["status"] != "NOT_EVALUABLE":
        rnd = {s: c["random"] for s, c in counts_by_seed.items()}
        jm = sum((jepa[s].fraction for s in SEEDS), Fraction(0)) / len(SEEDS)
        rm = sum((rnd[s].fraction for s in SEEDS), Fraction(0)) / len(SEEDS)
        report["random_control"] = {
            "random_mean_percent": float(rm * 100),
            "jepa_minus_random_percentage_points": float((jm - rm) * 100),
            "note": "reported separately, not part of the falsifier"}
    return report
