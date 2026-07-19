# COMP9517 26T2 Group Project — iNaturalist-2021 Fine-Grained Classification

Contract-driven repo (see `docs/CONTRACTS.md`). Two pipelines (traditional BoVW+SIFT+SVM,
deep CNN scratch+transfer) + two advanced studies (robustness, Grad-CAM), all behind one
`Method.predict_proba(df, image_transform)` interface. Plan: `docs/PROJECT_PLAN.md`.

Run everything as modules from the repo root (`python -m src...`).

## Setup
```bash
pip install -r requirements.txt
```

## 1. [A] Build the frozen subset (once; upload data/ to Drive)
```bash
python -m src.data.build_subset --inat_root /path/to/inat2021 --out data \
    --n_classes 500 --n_train 40 --n_val 10 --n_test 10 --seed 42
python -c "from src.common.manifest import validate_split; print(validate_split('data')[1])"  # M1 leakage check
```
Produces `data/classes_500.json`, `data/manifests/{train,val,test}.csv`, `data/images_256/`.

## 2. Methods (each writes results/<run>.json + predictions/<run>.npz)
```bash
python -m src.methods.transfer.run    --arch resnet50 --tag res50_ft   --epochs 30   # [D]
python -m src.methods.scratch.run     --arch resnet50 --tag res50_scr  --epochs 60   # [C]
python -m src.methods.traditional.bovw --k 512 --tag sift512                          # [B]
```
Deep ablations: `--no_aug`, `--arch resnet18|efficientnet_b0`.

## 3. [A] Aggregate + analysis
```bash
python -m src.eval.aggregate --results results                       # comparison table
python -m src.eval.analysis  --npz predictions/<run>.npz             # confusion + hardest pairs
python -m src.eval.evaluate  --npz predictions/<run>.npz             # recompute metrics from top-5
```

## 4. [E] Advanced (aim 28+)
```bash
python -m src.advanced.robustness.run --method deep --arch resnet50 \
    --ckpt data/checkpoints/transfer_resnet50__res50_ft/best.pt --tag res50_ft   # #2 robustness sweep
python -m src.advanced.gradcam.gradcam --arch resnet50 \
    --ckpt data/checkpoints/transfer_resnet50__res50_ft/best.pt --out figures/gradcam.png  # #1
```

## Submission
- video/ ≤10min, 5 present (name+face); report/ CVPR LaTeX ≤10pp; code ZIP ≤40MB
  (`.gitignore` already excludes models/images/npy per spec — `git archive` is submission-ready).
