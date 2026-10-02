import numpy as np
try:
    from moabb.datasets import BNCI2014_001 as BNCI2014001
except ImportError:
    from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery

TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7]
VAL_SUBJECTS = [8]
TEST_SUBJECTS = [9]
N_SAMPLES = 500
N_PATCHES = 10
PATCH_SIZE = 50
LABEL_MAP = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}


def load_subject(subject):
    """Load one subject, both sessions. Returns X (n,22,500) float32, y (n,) int64."""
    paradigm = MotorImagery(n_classes=4, fmin=4, fmax=40, tmin=0, tmax=2)
    X, y, meta = paradigm.get_data(dataset=BNCI2014001(), subjects=[subject])
    X = X[:, :, :N_SAMPLES].astype(np.float32)  # crop 501 -> 500 if needed
    y = np.array([LABEL_MAP[l] for l in y], dtype=np.int64)
    return X, y


def fit_norm_stats(X_train):
    """Per-channel mean/std from training subjects only. X_train: (n,22,500)."""
    mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1,22,1)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return mean, std


def apply_norm(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


def patchify(X):
    """(n,22,500) -> (n,10,22,50)."""
    n, c, t = X.shape
    assert t == N_SAMPLES, f"expected {N_SAMPLES} samples, got {t}"
    X = X.reshape(n, c, N_PATCHES, PATCH_SIZE)  # (n,22,10,50)
    return np.ascontiguousarray(X.transpose(0, 2, 1, 3))  # (n,10,22,50)


def load_splits(train_subjects=TRAIN_SUBJECTS, val_subjects=VAL_SUBJECTS,
                test_subjects=TEST_SUBJECTS):
    """Returns dict of patchified, normalized splits. Norm stats fit on train only."""
    all_ids = list(train_subjects) + list(val_subjects) + list(test_subjects)
    assert len(all_ids) == len(set(all_ids)), "subject leakage: overlap between splits"

    def load_group(subs):
        data = [load_subject(s) for s in subs]
        return (np.concatenate([d[0] for d in data]),
                np.concatenate([d[1] for d in data]))

    Xtr, ytr = load_group(train_subjects)
    mean, std = fit_norm_stats(Xtr)
    out = {"train": (patchify(apply_norm(Xtr, mean, std)), ytr)}
    for name, subs in [("val", val_subjects), ("test", test_subjects)]:
        X, y = load_group(subs)
        out[name] = (patchify(apply_norm(X, mean, std)), y)
    return out, (mean, std)
