"""Evaluate a trained CNN on the test split: top-1/top-5, macro P/R/F1,
confusion matrix, and inference timing. Prints the numbers the report needs.

  python evaluate.py --data ../data/subset500 --ckpt ../results/resnet50_pretrained/best.pt
"""
import argparse, json, os, time
import numpy as np, torch
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                             top_k_accuracy_score, accuracy_score, balanced_accuracy_score)
from deep_cnn import build_model
from dataset import get_loaders


@torch.no_grad()
def collect(model, loader, device, n_classes):
    model.eval()
    ys, probs = [], []
    t0 = time.time()
    for x, y in loader:
        out = model(x.to(device)).softmax(1).cpu().numpy()
        probs.append(out); ys.append(y.numpy())
    secs = time.time() - t0
    return np.concatenate(ys), np.concatenate(probs), secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    classes = ck["classes"]; n = len(classes)
    loaders, _ = get_loaders(args.data, batch_size=128)
    model = build_model(args.arch, n, pretrained=False).to(device)
    model.load_state_dict(ck["model"])

    y, prob, secs = collect(model, loaders["test"], device, n)
    pred = prob.argmax(1)
    top1 = accuracy_score(y, pred)
    top5 = top_k_accuracy_score(y, prob, k=5, labels=list(range(n)))
    bal = balanced_accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(range(n)))

    # hardest confused pairs (off-diagonal, symmetrised)
    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            c = off[i, j] + off[j, i]
            if c: pairs.append((c, classes[i], classes[j]))
    pairs.sort(reverse=True)

    res = {"top1": top1, "top5": top5, "balanced_acc": bal,
           "macro_precision": p, "macro_recall": r, "macro_f1": f1,
           "test_images": len(y), "infer_seconds": secs,
           "hardest_pairs": [{"count": int(c), "a": a, "b": b} for c, a, b in pairs[:15]]}
    print(json.dumps({k: v for k, v in res.items() if k != "hardest_pairs"}, indent=2))
    print("Hardest confused pairs:", res["hardest_pairs"][:5])

    out = args.out or os.path.join(os.path.dirname(args.ckpt), "eval.json")
    np.save(os.path.join(os.path.dirname(out), "confusion_matrix.npy"), cm)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("Saved ->", out, "(+ confusion_matrix.npy)")


if __name__ == "__main__":
    main()
