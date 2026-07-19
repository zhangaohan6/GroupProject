"""[A] Confusion matrix + hardest confused pairs (with genus/family) — 23-26 error analysis.

  python -m src.eval.analysis --npz predictions/<run>.npz --classes data/classes_500.json \
      --out figures/confusion.png
"""
import argparse, json
import numpy as np


def confusion_and_pairs(y_true, pred, classes=None, topk=20):
    from sklearn.metrics import confusion_matrix
    n = 500
    cm = confusion_matrix(y_true, pred, labels=list(range(n)))
    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            c = int(off[i, j] + off[j, i])
            if c:
                d = {"count": c, "a": i, "b": j}
                if classes:
                    ca, cb = classes["classes"][i], classes["classes"][j]
                    d.update({"a_species": ca.get("species"), "b_species": cb.get("species"),
                              "same_genus": ca.get("genus") == cb.get("genus"),
                              "same_family": ca.get("family") == cb.get("family")})
                pairs.append(d)
    pairs.sort(key=lambda x: -x["count"])
    return cm, pairs[:topk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--classes", default="data/classes_500.json")
    ap.add_argument("--out", default="figures/confusion.png")
    args = ap.parse_args()
    import os
    d = np.load(args.npz)
    pred = d["top5_idx"][:, 0]
    classes = json.load(open(args.classes)) if os.path.exists(args.classes) else None
    cm, pairs = confusion_and_pairs(d["y_true"], pred, classes)
    print("Hardest confused pairs (same_genus flagged):")
    for p in pairs[:10]:
        print(" ", p)
    try:
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        plt.figure(figsize=(6, 5)); plt.imshow(np.log1p(cm), cmap="magma")
        plt.title("Confusion (log)"); plt.colorbar(); plt.tight_layout(); plt.savefig(args.out, dpi=150)
        print("Saved", args.out)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
