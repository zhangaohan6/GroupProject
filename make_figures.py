#!/usr/bin/env python3
"""Generate all report figures from results/ + predictions/ into report/figures/.

Figures produced (referenced by report/main.tex):
  1. training_curves.png   — D transfer loss/acc vs epoch (train+val). Auto-includes
                             C scratch history if results/scratch_*_history.json exists.
  2. confusion_matrix.png  — 500x500 confusion (row-normalised) for the pretrained model.
  3. hardest_pairs.png     — top-15 most-confused ordered species pairs (bar chart).
  4. robustness_curves.png — top-1 & macro-F1 vs severity, one line per corruption.

Run:  python3 make_figures.py
Idempotent; safe to re-run after C finishes (re-picks up new history/predictions).
"""
from __future__ import annotations
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
PRED = os.path.join(ROOT, "predictions")
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)

CORRUPTIONS = ["jpeg", "brightness", "gaussian_noise", "gaussian_blur", "motion_blur"]
SEVERITIES = [1, 2, 3, 4, 5]


def _load(path):
    with open(path) as f:
        return json.load(f)


def fig_training_curves():
    """Loss/acc vs epoch for every *_history.json found (D always; C when ready)."""
    hist_files = sorted(glob.glob(os.path.join(RES, "*__history.json")))
    hist_files = [h for h in hist_files if "smoke" not in h]  # drop smoke runs
    if not hist_files:
        print("  [skip] no history json"); return
    fig, (axL, axA) = plt.subplots(1, 2, figsize=(10, 4))
    for hf in hist_files:
        tag = re.sub(r"__history\.json$", "", os.path.basename(hf))
        tag = tag.replace("transfer_resnet50__", "pretrained-").replace(
            "scratch_resnet50__", "scratch-")
        h = _load(hf)["history"]
        ep = [r["epoch"] for r in h]
        axL.plot(ep, [r["train_loss"] for r in h], "-", label=f"{tag} train")
        axL.plot(ep, [r["val_loss"] for r in h], "--", label=f"{tag} val")
        axA.plot(ep, [r["train_acc"] for r in h], "-", label=f"{tag} train")
        axA.plot(ep, [r["val_acc"] for r in h], "--", label=f"{tag} val")
    axL.set(xlabel="epoch", ylabel="loss", title="Loss")
    axA.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    for ax in (axL, axA):
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG, "training_curves.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [ok] {out}")


def _clean_preds():
    """Return (y_true, y_pred) for the clean pretrained test predictions."""
    p = os.path.join(PRED, "transfer_resnet50__res50_ft__test.npz")
    if not os.path.exists(p):
        return None, None
    d = np.load(p)
    return d["y_true"], d["top5_idx"][:, 0]


def fig_confusion_and_pairs():
    y_true, y_pred = _clean_preds()
    if y_true is None:
        print("  [skip] no clean transfer predictions npz"); return
    n = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    # row-normalised confusion matrix
    rown = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(rown, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set(xlabel="predicted class", ylabel="true class",
           title=f"Confusion matrix ({n} classes, row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = os.path.join(FIG, "confusion_matrix.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [ok] {out}")

    # hardest confused ordered pairs (off-diagonal mass)
    off = cm.copy(); np.fill_diagonal(off, 0)
    idx = np.argsort(off.ravel())[::-1][:15]
    pairs = [(i // n, i % n, off[i // n, i % n]) for i in idx if off.ravel()[i] > 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"{t}→{p}" for t, p, _ in pairs]
    ax.barh(range(len(pairs)), [c for *_, c in pairs], color="#c0392b")
    ax.set_yticks(range(len(pairs))); ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set(xlabel="# misclassified", title="Top-15 hardest confused class pairs (true→pred)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = os.path.join(FIG, "hardest_pairs.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [ok] {out}")


def fig_robustness_curves():
    fig, (ax1, axF) = plt.subplots(1, 2, figsize=(10, 4))
    clean = _load(os.path.join(RES, "transfer_resnet50__res50_ft__test.json"))["metrics"]
    for corr in CORRUPTIONS:
        t1 = [clean["top1"]]; mf = [clean["macro_f1"]]; xs = [0]
        for s in SEVERITIES:
            f = os.path.join(RES, f"scratch_resnet50__res50_ft__{corr}_s{s}__test.json")
            if not os.path.exists(f):
                continue
            m = _load(f)["metrics"]
            xs.append(s); t1.append(m["top1"]); mf.append(m["macro_f1"])
        ax1.plot(xs, t1, "-o", ms=3, label=corr)
        axF.plot(xs, mf, "-o", ms=3, label=corr)
    ax1.set(xlabel="severity (0=clean)", ylabel="top-1", title="Top-1 vs corruption severity")
    axF.set(xlabel="severity (0=clean)", ylabel="macro-F1", title="Macro-F1 vs corruption severity")
    for ax in (ax1, axF):
        ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_ylim(0, 0.75)
    fig.tight_layout()
    out = os.path.join(FIG, "robustness_curves.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [ok] {out}")


if __name__ == "__main__":
    print("generating figures ->", FIG)
    fig_training_curves()
    fig_confusion_and_pairs()
    fig_robustness_curves()
    print("done.")
