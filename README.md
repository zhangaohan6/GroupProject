# COMP9517 26T2 Group Project — iNaturalist-2021 Fine-Grained Species Classification

Two genuinely different pipelines (traditional BoVW+SIFT+SVM and a deep CNN), plus
two advanced studies (test-time robustness + Grad-CAM). See `PLAN.md` for the full plan,
timeline, marking targets, and team split.

## Setup
```bash
pip install -r requirements.txt
```

## 1. Build the graded subset (≥500 classes, 40/10/10 per class, fixed seed)
Download iNat2021 mini (`train_mini`, `val` + their `.json`) from
https://github.com/visipedia/inat_comp/tree/master/2021 , then:
```bash
python src/build_subset.py --inat_root /path/to/inat2021 --out data \
    --n_classes 500 --n_train 40 --n_val 10 --n_test 10 --seed 42
```
Produces `data/{train,val,test}/<class>/...` + `manifest.json` (reproducible).

## 2. Deep pipeline (pretrained + from-scratch, required)
```bash
python src/deep_cnn.py --data data --arch resnet50 --pretrained \
    --epochs 30 --run_id resnet50_pretrained_42 --out results
python src/deep_cnn.py --data data --arch resnet50 \
    --epochs 60 --run_id resnet50_scratch_42 --out results
```
Ablations: `--no_aug` (augmentation off), `--arch resnet18|efficientnet_b0` (architecture).

## 3. Traditional pipeline
```bash
python src/traditional_bovw.py --data data --k 512 --run_id bovw_sift512_42 --out results
```

## 4. Aggregate results + leakage check
Each runner writes `results/<run_id>/result.json` via `common.save_result`. Then:
```bash
python src/evaluate.py --results results            # comparison table across runs
python src/evaluate.py --data data --leakage_check  # assert train/val/test disjoint
```

## 5. Advanced studies (aim for 28+)
```bash
python src/robustness.py --data data --ckpt results/resnet50_pretrained_42/best.pt  # #2
python src/gradcam.py    --data data --ckpt results/resnet50_pretrained_42/best.pt  # #1
```

## Colab
Store `data/` in Drive, copy to local disk each session, train on T4. Save checkpoints
(`last.pt`) to resume. Subset of 500 classes × 60 imgs is a few hundred MB — comfortable.

## Submission (Week 10)
- **video/**: ≤10 min MP4, all 5 present (name+face in corner), incl. demo
- **report/**: CVPR LaTeX template, PDF, ≤10 pages
- **code**: ZIP ≤40MB — code only, no trained models / images / result images
