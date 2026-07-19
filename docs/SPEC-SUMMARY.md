# 规格 / 评分要点摘录(权威以官方 PDF 为准)

## 截止 & 权重
- 30% of course · Due **Fri 7 Aug 2026 22:00 AET** · team of 5

## 数据(评分实验最低)
- iNat2021 mini (train_mini split). ≥500 classes (random), ≥50 train img/class (40/10),
  ≥10 test/class from official **val** split (official test labels unreleased).
- Report exact classes + counts + seed. No leakage. Only iNat2021 data for graded eval
  (except a self-designed advanced topic).

## 方法(必须)
- ≥2 genuinely different complete methods: (1) traditional handcrafted+classical
  (e.g. BoVW-SIFT + SVM), (2) deep CNN with BOTH from-scratch AND ImageNet-pretrained.

## 指标
- top-1, top-5, overall acc, (balanced acc), macro P/R/F1. Emphasis on macro-F1 +
  confusion matrix + hardest confused pairs. Report train/test time vs performance.

## 分档
- 23–26 comprehensive: variety + ablations (one factor at a time) + training curves
  (loss/acc) + error analysis.
- 26–27: one advanced direction in depth. 28+: two or more.

## Advanced 方向(官方点名 4 个)
1. Grad-CAM explainability (organism vs background; correct vs wrong; confusable pairs)
2. Test-time degradation robustness — degrade ONLY test set, ≥4 corruption types × several
   severities; top-1 & macro-F1 vs severity curves; model fixed, no retrain.
3. Effect of #classes — scale 500→more; acc/macro-F1 vs granularity; document subsets+seed.
4. Continual (class-incremental) learning — catastrophic forgetting + replay (De Lange /
   Chaudhry metrics); can restrict to 100–200 classes.
   Others: bilinear/2nd-order pooling, long-tail, metric/few-shot, weakly-sup segmentation.

## 交付
- Video ≤10 min MP4 720p/1080p ≤100MB, all 5 present (name+face corner) + demo.
- Report: **CVPR LaTeX** (\usepackage[pagenumbers]{cvpr}), PDF, ≤10 pages (+refs), ≤10MB.
  Wrong template may score 0.
- Code: ZIP ≤40MB, own code only, README, no models/images.
- Week 11 anonymous peer contribution survey (can moderate individual marks).
