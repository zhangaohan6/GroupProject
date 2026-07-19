"""Shared base for the whole team — see docs/CONTRACTS.md. Everyone imports from here.

- load_image(): the ONE image entry point (RGB PIL). Never cv2.imread/Image.open elsewhere.
- Method: abstract base every pipeline implements (predict_proba with image_transform).
- read_manifest / save_result / compute_metrics: unified data + result + metric contracts.
"""
import json, os
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image

N_CLASSES = 500


# ---- image entry point (CONTRACTS §2.2) -------------------------------------
def load_image(path):
    """Always RGB PIL.Image. Do not use cv2.imread / Image.open anywhere else."""
    return Image.open(path).convert("RGB")


# ---- manifest (CONTRACTS §2.3) ----------------------------------------------
def read_manifest(csv_path):
    """Returns list of dicts: [{'path':..., 'class_id':int, 'split':...}, ...]."""
    import csv
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["class_id"] = int(r["class_id"])
    return rows


def load_classes(json_path):
    with open(json_path) as f:
        return json.load(f)


# ---- method interface (CONTRACTS §3) ----------------------------------------
class Method(ABC):
    name: str = "method"

    @abstractmethod
    def fit(self, train_df, val_df):
        ...

    @abstractmethod
    def predict_proba(self, df, image_transform=None) -> np.ndarray:
        """(len(df), N_CLASSES) row-stochastic. image_transform: callable(PIL)->PIL
        applied per image after load_image, before the model. Default None."""
        ...


# ---- metrics + result (CONTRACTS §4,§5) -------------------------------------
def compute_metrics(y_true, probs, class_names=None):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 precision_recall_fscore_support, top_k_accuracy_score,
                                 confusion_matrix)
    y_true = np.asarray(y_true)
    pred = probs.argmax(1)
    labels = list(range(N_CLASSES))
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="macro",
                                                  labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=labels)
    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = []
    for i in range(N_CLASSES):
        for j in range(i + 1, N_CLASSES):
            c = int(off[i, j] + off[j, i])
            if c:
                a = class_names[i] if class_names else i
                b = class_names[j] if class_names else j
                pairs.append({"count": c, "a": a, "b": b})
    pairs.sort(key=lambda d: -d["count"])
    return {
        "top1": float(accuracy_score(y_true, pred)),
        "top5": float(top_k_accuracy_score(y_true, probs, k=5, labels=labels)),
        "overall_acc": float(accuracy_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "macro_p": float(p), "macro_r": float(r), "macro_f1": float(f1),
        "hardest_pairs": pairs[:15],
    }, cm


def save_result(results_dir, run_id, method, y_true, probs,
                train_seconds=None, test_seconds=None, class_names=None):
    """Writes results/<run_id>/result.json + topk.npz (CONTRACTS §4)."""
    out = os.path.join(results_dir, run_id)
    os.makedirs(out, exist_ok=True)
    metrics, cm = compute_metrics(y_true, probs, class_names)
    result = {"run_id": run_id, "method": method,
              "train_seconds": train_seconds, "test_seconds": test_seconds, **metrics}
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    order = np.argsort(-probs, 1)[:, :5]
    top5_prob = np.take_along_axis(probs, order, 1)
    np.savez_compressed(os.path.join(out, "topk.npz"),
                        top5_idx=order.astype(np.int32),
                        top5_prob=top5_prob.astype(np.float32),
                        y_true=np.asarray(y_true, np.int32))
    np.save(os.path.join(out, "confusion_matrix.npy"), cm)
    return result
