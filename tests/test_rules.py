import os
import sys
import unittest
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.selection import (SEEDS, Accuracy, Candidate, candidate_grid, falsifier,
                           loss_trend_flag, select_on_s8)


def make(model, seed, score_fn, total=100):
    return [Candidate(model, seed, k[0], k[1], k[2], Accuracy(score_fn(k), total))
            for k in candidate_grid(model, seed)]


def accs(correct, total=100):
    return {s: Accuracy(correct, total) for s in SEEDS}


class GridTests(unittest.TestCase):
    def test_grid_sizes(self):
        sizes = {m: len(candidate_grid(m, 42)) for m in ("jepa", "baseline", "random")}
        self.assertEqual(sizes, {"jepa": 20, "baseline": 20, "random": 4})
        self.assertEqual(3 * sum(sizes.values()), 132)

    def test_random_has_zero_checkpoint_sentinel(self):
        self.assertTrue(all(k[0] == 0 for k in candidate_grid("random", 7)))


class SelectionTests(unittest.TestCase):
    def test_full_tie_picks_earliest_epoch_smallest_lr_fewest_probe_epochs(self):
        self.assertEqual(select_on_s8("jepa", 42, make("jepa", 42, lambda k: 50)).key,
                         (10, Decimal("0.001"), 50))
        self.assertEqual(select_on_s8("baseline", 42, make("baseline", 42, lambda k: 50)).key,
                         (10, Decimal("0.00025"), 0))
        self.assertEqual(select_on_s8("random", 42, make("random", 42, lambda k: 50)).key,
                         (0, Decimal("0.001"), 50))

    def test_epoch_ties_before_lr(self):
        def score(k):
            return 60 if k in ((20, Decimal("0.001"), 50), (10, Decimal("0.01"), 100)) else 40
        self.assertEqual(select_on_s8("jepa", 42, make("jepa", 42, score)).key,
                         (10, Decimal("0.01"), 100))

    def test_lr_ties_before_probe_epochs(self):
        def score(k):
            return 60 if k in ((10, Decimal("0.001"), 100), (10, Decimal("0.01"), 50)) else 40
        self.assertEqual(select_on_s8("jepa", 42, make("jepa", 42, score)).key,
                         (10, Decimal("0.001"), 100))

    def test_higher_score_beats_earlier_epoch(self):
        def score(k):
            return 70 if k == (50, Decimal("0.01"), 100) else 60
        self.assertEqual(select_on_s8("jepa", 42, make("jepa", 42, score)).key,
                         (50, Decimal("0.01"), 100))

    def test_incomplete_grid_rejected(self):
        with self.assertRaises(ValueError):
            select_on_s8("jepa", 42, make("jepa", 42, lambda k: 50)[:-1])

    def test_duplicate_rejected(self):
        cands = make("jepa", 42, lambda k: 50)
        with self.assertRaises(ValueError):
            select_on_s8("jepa", 42, cands + [cands[0]])

    def test_mixed_trial_counts_rejected(self):
        cands = make("jepa", 42, lambda k: 50)
        cands[0] = Candidate("jepa", 42, *cands[0].key, Accuracy(50, 99))
        with self.assertRaises(ValueError):
            select_on_s8("jepa", 42, cands)

    def test_mixed_seed_rejected(self):
        cands = make("jepa", 42, lambda k: 50)
        with self.assertRaises(ValueError):
            select_on_s8("jepa", 7, cands)


class FalsifierTests(unittest.TestCase):
    def test_f1_inclusive_boundary(self):
        self.assertTrue(falsifier(accs(27), accs(27))["f1"])
        self.assertFalse(falsifier(accs(28), accs(28))["f1"])

    def test_exact_fractions_at_boundary(self):
        self.assertTrue(falsifier(accs(155, 576), accs(155, 576))["f1"])
        self.assertFalse(falsifier(accs(156, 576), accs(156, 576))["f1"])

    def test_below_chance_fires(self):
        out = falsifier(accs(20), accs(25))
        self.assertEqual(out["status"], "FALSIFIER_FIRED")
        self.assertTrue(out["f1"])
        self.assertFalse(out["f2"])

    def test_f2_strict_boundary(self):
        self.assertFalse(falsifier(accs(30), accs(40))["f2"])
        self.assertEqual(falsifier(accs(30), accs(40))["status"], "FALSIFIER_NOT_FIRED")
        self.assertTrue(falsifier(accs(30), accs(41))["f2"])

    def test_f1_and_f2_both_reported(self):
        out = falsifier(accs(27), accs(38))
        self.assertTrue(out["f1"] and out["f2"])

    def test_uses_seed_mean_not_each_seed(self):
        jepa = {42: Accuracy(20, 100), 7: Accuracy(40, 100), 123: Accuracy(40, 100)}
        out = falsifier(jepa, accs(40))
        self.assertFalse(out["f1"])

    def test_missing_seed_not_evaluable(self):
        jepa = {42: Accuracy(40, 100), 7: Accuracy(40, 100)}
        self.assertEqual(falsifier(jepa, accs(40))["status"], "NOT_EVALUABLE")

    def test_missing_score_not_evaluable(self):
        jepa = {42: Accuracy(40, 100), 7: Accuracy(40, 100), 123: None}
        self.assertEqual(falsifier(jepa, accs(40))["status"], "NOT_EVALUABLE")

    def test_nonfinite_score_not_evaluable(self):
        jepa = {42: Accuracy(40, 100), 7: Accuracy(40, 100), 123: float("nan")}
        self.assertEqual(falsifier(jepa, accs(40))["status"], "NOT_EVALUABLE")

    def test_mixed_trial_counts_not_evaluable(self):
        self.assertEqual(falsifier(accs(40, 100), accs(40, 99))["status"], "NOT_EVALUABLE")

    def test_non_firing_is_not_positive(self):
        out = falsifier(accs(40), accs(41))
        self.assertEqual(out["status"], "FALSIFIER_NOT_FIRED")
        self.assertIn("not positive evidence", out["interpretation"])

    def test_sample_std_convention(self):
        jepa = {42: Accuracy(30, 100), 7: Accuracy(40, 100), 123: Accuracy(50, 100)}
        out = falsifier(jepa, accs(60))
        self.assertAlmostEqual(out["jepa_sample_std_percentage_points"], 10.0, places=6)


class TrendFlagTests(unittest.TestCase):
    def test_flat_losses_not_flagged(self):
        self.assertFalse(loss_trend_flag([1.0] * 50))

    def test_decline_over_two_percent_flagged(self):
        self.assertTrue(loss_trend_flag([1.0] * 40 + [0.97] * 10))

    def test_exactly_two_percent_not_flagged(self):
        self.assertFalse(loss_trend_flag([1.0] * 40 + [0.98] * 10))

    def test_wrong_length_rejected(self):
        with self.assertRaises(ValueError):
            loss_trend_flag([1.0] * 49)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            loss_trend_flag([1.0] * 49 + [float("nan")])


if __name__ == "__main__":
    unittest.main()
