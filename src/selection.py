import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from itertools import product

SEEDS = (42, 7, 123)
CHECKPOINTS = (10, 20, 30, 40, 50)
PROBE_RATES = (Decimal("0.001"), Decimal("0.01"))
PROBE_EPOCHS = (50, 100)
BASELINE_RATES = tuple(Decimal(x) for x in ("0.00025", "0.0005", "0.001", "0.002"))
MODELS = ("jepa", "baseline", "random")


def _int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _rate(value):
    if isinstance(value, bool):
        raise ValueError("learning rate must be finite and positive")
    try:
        rate = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("invalid learning rate") from exc
    if not rate.is_finite() or rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    return rate


def _model_seed(model, seed):
    if model not in MODELS:
        raise ValueError("unknown model")
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("seed must be exactly one of 42, 7, 123")


@dataclass(frozen=True)
class Accuracy:
    correct: int
    total: int

    def __post_init__(self):
        _int(self.correct, "correct")
        _int(self.total, "total", 1)
        if self.correct > self.total:
            raise ValueError("correct exceeds total")

    @property
    def fraction(self):
        return Fraction(self.correct, self.total)


@dataclass(frozen=True)
class Candidate:
    model: str
    seed: int
    checkpoint_epoch: int
    learning_rate: Decimal
    probe_epochs: int
    s8: Accuracy

    def __post_init__(self):
        _model_seed(self.model, self.seed)
        _int(self.checkpoint_epoch, "checkpoint_epoch")
        _int(self.probe_epochs, "probe_epochs")
        object.__setattr__(self, "learning_rate", _rate(self.learning_rate))
        if not isinstance(self.s8, Accuracy):
            raise ValueError("S8 score must be exact Accuracy counts")

    @property
    def key(self):
        return self.checkpoint_epoch, self.learning_rate, self.probe_epochs


def candidate_grid(model, seed, checkpoints=CHECKPOINTS, probe_rates=PROBE_RATES,
                   probe_epochs=PROBE_EPOCHS, baseline_rates=BASELINE_RATES):
    _model_seed(model, seed)
    if model == "jepa":
        return tuple(product(checkpoints, probe_rates, probe_epochs))
    if model == "baseline":
        return tuple((e, lr, 0) for e, lr in product(checkpoints, baseline_rates))
    return tuple((0, lr, e) for lr, e in product(probe_rates, probe_epochs))


def select_on_s8(model, seed, candidates, grid=None):
    expected = set(grid if grid is not None else candidate_grid(model, seed))
    if any(not isinstance(c, Candidate) for c in candidates):
        raise ValueError("every candidate must be a Candidate")
    if any(c.model != model or c.seed != seed for c in candidates):
        raise ValueError("mixed model or seed in selection input")
    keys = [c.key for c in candidates]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate candidate")
    if set(keys) != expected:
        raise ValueError("candidate grid is incomplete or outside the planned grid")
    if len({c.s8.total for c in candidates}) != 1:
        raise ValueError("all candidates must be scored over the same S8 trial count")
    return min(candidates, key=lambda c: (-c.s8.fraction, *c.key))


def _validate_primary(scores, name):
    if not isinstance(scores, dict):
        raise ValueError(f"{name}: scores must be a mapping")
    if any(type(seed) is not int for seed in scores) or set(scores) != set(SEEDS):
        raise ValueError(f"{name}: exactly seeds 42, 7 and 123 are required")
    if any(not isinstance(score, Accuracy) for score in scores.values()):
        raise ValueError(f"{name}: missing or invalid trial counts")


def _exact(value):
    return f"{value.numerator}/{value.denominator}"


def falsifier(jepa, baseline):
    try:
        _validate_primary(jepa, "jepa")
        _validate_primary(baseline, "baseline")
        totals = {s.total for group in (jepa, baseline) for s in group.values()}
        if len(totals) != 1:
            raise ValueError("all six scores must use the same S9 trial count")
    except (ValueError, TypeError, AttributeError) as exc:
        return {"status": "NOT_EVALUABLE", "f1": None, "f2": None, "reason": str(exc)}
    jmean = sum((jepa[s].fraction for s in SEEDS), Fraction(0)) / len(SEEDS)
    bmean = sum((baseline[s].fraction for s in SEEDS), Fraction(0)) / len(SEEDS)
    gap = bmean - jmean
    f1, f2 = jmean <= Fraction(27, 100), gap > Fraction(10, 100)

    def sample_std(group, mean):
        variance = sum(((group[s].fraction - mean) ** 2 for s in SEEDS), Fraction(0)) / (len(SEEDS) - 1)
        return math.sqrt(float(variance)) * 100

    return {
        "status": "FALSIFIER_FIRED" if f1 or f2 else "FALSIFIER_NOT_FIRED",
        "f1": f1, "f2": f2,
        "jepa_mean_fraction": _exact(jmean), "baseline_mean_fraction": _exact(bmean),
        "baseline_minus_jepa_fraction": _exact(gap),
        "jepa_mean_percent": float(jmean * 100), "baseline_mean_percent": float(bmean * 100),
        "baseline_minus_jepa_percentage_points": float(gap * 100),
        "jepa_sample_std_percentage_points": sample_std(jepa, jmean),
        "baseline_sample_std_percentage_points": sample_std(baseline, bmean),
        "std_convention": "sample standard deviation (ddof=1), display only",
        "interpretation": "A falsifier that does not fire is not positive evidence. Seed variation is optimization variation on one held-out subject.",
    }


def loss_trend_flag(losses):
    if len(losses) != 50:
        raise ValueError("exactly 50 epoch losses required")
    converted = []
    for value in losses:
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, Fraction)):
            raise ValueError("loss must be a finite nonnegative number")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("loss must be finite")
        try:
            number = Fraction(str(value))
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            raise ValueError("loss must be finite") from exc
        if number < 0:
            raise ValueError("loss must be nonnegative")
        converted.append(number)
    earlier, later = sum(converted[30:40]), sum(converted[40:50])
    return later < earlier * Fraction(98, 100)
