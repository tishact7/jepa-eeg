import json
import os
import shutil
import sys
import tempfile
import unittest
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import src.dataset as dataset
import src.orchestrate as orch
from src.evaluate import extract_features
from src.models import Encoder
from src.selection import SEEDS, Accuracy

SMALL = orch.Plan(epochs=2, checkpoints=(1, 2),
                  probe_rates=(Decimal("0.001"), Decimal("0.01")),
                  probe_epochs=(2, 3),
                  baseline_rates=(Decimal("0.001"), Decimal("0.002")),
                  batch_size=8)


def synthetic_arrays(seed, n):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 10, 22, 50).astype(np.float32)
    y = (np.arange(n) % 4).astype(np.int64)
    return X, y


def synthetic_data():
    Xtr, ytr = synthetic_arrays(1, 32)
    Xval, yval = synthetic_arrays(2, 16)
    return {"Xtr": Xtr, "ytr": ytr, "Xval": Xval, "yval": yval}


class SubjectSpy:
    def __init__(self):
        self.calls = []

    def __call__(self, subject):
        self.calls.append(subject)
        rng = np.random.RandomState(subject)
        X = rng.randn(12, 22, 500).astype(np.float32)
        y = (np.arange(12) % 4).astype(np.int64)
        return X, y


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.orig_loader = dataset.load_subject

    def tearDown(self):
        dataset.load_subject = self.orig_loader
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_selected(self):
        return orch.run_seed(42, synthetic_data(), self.tmp, "cpu", plan=SMALL, make_tsne=False)

    def test_selection_path_never_loads_s9(self):
        spy = SubjectSpy()
        dataset.load_subject = spy
        data, (mean, std) = dataset.load_train_val()
        self.assertEqual(sorted(spy.calls), [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertNotIn(9, spy.calls)
        self.assertEqual(data["Xtr"].shape, (84, 10, 22, 50))
        record = orch.run_seed(42, {k: v[:16] for k, v in data.items()}, self.tmp, "cpu",
                               plan=SMALL, make_tsne=False)
        self.assertEqual(record["status"], "SELECTED")
        self.assertFalse(record["s9_accessed"])
        self.assertNotIn(9, spy.calls)

    def test_s9_loader_refuses_without_selected_records(self):
        spy = SubjectSpy()
        dataset.load_subject = spy
        mean, std = np.zeros((1, 22, 1), np.float32), np.ones((1, 22, 1), np.float32)
        with self.assertRaises(PermissionError):
            dataset.load_test(mean, std, [])
        with self.assertRaises(PermissionError):
            dataset.load_test(mean, std, [{"seed": s, "status": "NOT_EVALUABLE"} for s in SEEDS])
        self.assertNotIn(9, spy.calls)
        X, y = dataset.load_test(mean, std, [{"seed": s, "status": "SELECTED"} for s in SEEDS])
        self.assertEqual(spy.calls, [9])
        self.assertEqual(X.shape[1:], (10, 22, 50))

    def test_run_seed_records_complete_grid_and_selection(self):
        record = self.run_selected()
        self.assertEqual(record["status"], "SELECTED")
        self.assertEqual(record["s8_evaluations"], 8 + 4 + 4)
        by_model = {}
        for c in record["candidates"]:
            by_model.setdefault(c["model"], []).append(c)
        self.assertEqual({m: len(v) for m, v in by_model.items()},
                         {"jepa": 8, "baseline": 4, "random": 4})
        path = os.path.join(self.tmp, "selection_record_seed42.json")
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk["selected"], record["selected"])
        for m, sel in record["selected"].items():
            best = max(c["s8_correct"] for c in by_model[m])
            self.assertEqual(sel["s8_correct"], best)

    def test_final_eval_uses_recorded_selected_candidate(self):
        record = self.run_selected()
        Xtest, ytest = synthetic_arrays(9, 20)
        counts = orch.final_counts(record, Xtest, ytest, "cpu")
        j = record["selected"]["jepa"]
        ck = torch.load(j["checkpoint"], map_location="cpu")
        enc = Encoder()
        enc.load_state_dict(ck["models"]["encoder"])
        feats = extract_features(enc, Xtest, "cpu")
        probe = torch.load(j["probe"], map_location="cpu")
        logits = ((feats - probe["mu"]) / probe["sd"]) @ probe["state"]["weight"].T + probe["state"]["bias"]
        expected = int((logits.argmax(1) == torch.as_tensor(ytest)).sum().item())
        self.assertEqual(counts["jepa"], Accuracy(expected, 20))
        self.assertEqual(counts["baseline"].total, 20)
        self.assertEqual(counts["random"].total, 20)

    def test_final_eval_rejects_changed_artifact(self):
        record = self.run_selected()
        Xtest, ytest = synthetic_arrays(9, 20)
        with open(record["selected"]["jepa"]["probe"], "ab") as f:
            f.write(b"x")
        with self.assertRaises(ValueError):
            orch.final_counts(record, Xtest, ytest, "cpu")

    def test_failed_candidate_makes_run_not_evaluable(self):
        real = orch.fit_probe
        state = {"n": 0}

        def flaky(*a, **kw):
            out = real(*a, **kw)
            state["n"] += 1
            if state["n"] == 3:
                out["finite"] = False
            return out

        orch.fit_probe = flaky
        try:
            record = self.run_selected()
        finally:
            orch.fit_probe = real
        self.assertEqual(record["status"], "NOT_EVALUABLE")
        self.assertNotIn("selected", record)
        self.assertTrue(record["failures"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "seed42", "selected")))
        self.assertFalse(record["s9_accessed"])

    def test_diverged_baseline_makes_run_not_evaluable(self):
        real = orch.train_baseline

        def diverged(*a, **kw):
            out = real(*a, **kw)
            out["finite"] = False
            return out

        orch.train_baseline = diverged
        try:
            record = self.run_selected()
        finally:
            orch.train_baseline = real
        self.assertEqual(record["status"], "NOT_EVALUABLE")
        self.assertNotIn("selected", record)

    def test_final_report_needs_all_seeds(self):
        record = self.run_selected()
        Xtest, ytest = synthetic_arrays(9, 20)
        counts = {42: orch.final_counts(record, Xtest, ytest, "cpu")}
        report = orch.build_final_report([record], counts)
        self.assertEqual(report["falsifier"]["status"], "NOT_EVALUABLE")
        self.assertNotIn("random_control", report)

    def test_random_control_reported_outside_falsifier(self):
        record = self.run_selected()
        Xtest, ytest = synthetic_arrays(9, 20)
        one = orch.final_counts(record, Xtest, ytest, "cpu")
        counts = {s: one for s in SEEDS}
        records = [dict(record, seed=s) for s in SEEDS]
        report = orch.build_final_report(records, counts)
        self.assertIn("random_control", report)
        self.assertNotIn("random", json.dumps(report["falsifier"]))


if __name__ == "__main__":
    unittest.main()
