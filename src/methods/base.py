"""[A] Method ABC + run/save helpers — CONTRACTS §3.1, §4. B/C/D/E build on this."""
import json
import os
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
import numpy as np

N_CLASSES = 500


class Method(ABC):
    """Unified interface for every pipeline (traditional / scratch / transfer)."""
    name: str = "method"     # e.g. "transfer_resnet50"
    config: dict = {}        # all hyperparameters, copied verbatim into result JSON

    @abstractmethod
    def fit(self, train_df, val_df):
        """Fit on train_df; tune/early-stop/select on val_df. NEVER touch test here."""

    @abstractmethod
    def predict_proba(self, df, image_transform=None) -> np.ndarray:
        """(len(df), 500) float32, row order == df, col j == class_id j, each row softmax-summed.
        image_transform: callable(ndarray HWC uint8 RGB)->same, applied after load_image,
        before this method's own preprocessing (CONTRACTS §3.2)."""

    @abstractmethod
    def save(self, path): ...

    @abstractmethod
    def load(self, path): ...


# ---- result contract (CONTRACTS §4) -----------------------------------------
def run_name(method, tag, split, degradation=None):
    base = f"{method}__{tag}"
    if degradation:
        base += f"__{degradation['type']}_s{degradation['severity']}"
    return f"{base}__{split}"


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


def save_run(repo_root, method, tag, split, y_true, probs, config,
             train_seconds=None, predict_seconds=None, degradation=None):
    """Writes results/<run>.json + predictions/<run>.npz (CONTRACTS §4.2/4.3)."""
    from src.eval.evaluate import metrics_from_probs
    run = run_name(method, tag, split, degradation)
    y_true = np.asarray(y_true)
    met = metrics_from_probs(y_true, probs)
    result = {
        "run": run, "method": method, "tag": tag, "split": split,
        "degradation": degradation, "config": config, "metrics": met,
        "timing": {"train_seconds": train_seconds, "predict_seconds": predict_seconds},
        "n_samples": int(len(y_true)), "n_classes": N_CLASSES, "seed": 42,
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone(timedelta(hours=10))).isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.join(repo_root, "results"), exist_ok=True)
    os.makedirs(os.path.join(repo_root, "predictions"), exist_ok=True)
    with open(os.path.join(repo_root, "results", f"{run}.json"), "w") as f:
        json.dump(result, f, indent=2)
    order = np.argsort(-probs, 1)[:, :5]
    np.savez_compressed(os.path.join(repo_root, "predictions", f"{run}.npz"),
                        top5_idx=order.astype(np.int16),
                        top5_prob=np.take_along_axis(probs, order, 1).astype(np.float32),
                        y_true=y_true.astype(np.int16))
    return result
