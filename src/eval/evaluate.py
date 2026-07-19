"""[A] Metrics from a probability matrix — CONTRACTS §4.2. Also reruns metrics from a
saved predictions/*.npz (top-5 is enough to recompute everything, CONTRACTS §4.3).

  python -m src.eval.evaluate --npz predictions/<run>.npz
"""
import argparse
import numpy as np

N_CLASSES = 500


def metrics_from_probs(y_true, probs):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 precision_recall_fscore_support, top_k_accuracy_score)
    y_true = np.asarray(y_true); pred = probs.argmax(1); labels = list(range(N_CLASSES))
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="macro",
                                                  labels=labels, zero_division=0)
    return {
        "top1": float(accuracy_score(y_true, pred)),
        "top5": float(top_k_accuracy_score(y_true, probs, k=5, labels=labels)),
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }


def metrics_from_topk(npz_path):
    """Reconstruct a (N,500) sparse prob matrix from top-5 and score it."""
    d = np.load(npz_path)
    N = len(d["y_true"])
    probs = np.zeros((N, N_CLASSES), np.float32)
    np.put_along_axis(probs, d["top5_idx"].astype(int), d["top5_prob"], 1)
    return metrics_from_probs(d["y_true"], probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    args = ap.parse_args()
    import json
    print(json.dumps(metrics_from_topk(args.npz), indent=2))


if __name__ == "__main__":
    main()
