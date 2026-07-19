# CONTRACTS.md — 技术契约(**以本文件为准**)

> 5 人并行的地基。**7/20 kickoff 当天定死,之后不改**。改任何一条要全队同意并记进变更日志。
> 配套:[PROJECT_PLAN.md](PROJECT_PLAN.md)(总纲)· [M0_KICKOFF.md](M0_KICKOFF.md)(启动)

## 1. 目录 & 命名
```
data/            # gitignore,不入库
  classes_500.json          # 冻结的类映射(A 产出,永不变)
  manifests/{train,val,test}.csv
  images/<class_id>/<img>   # 256px 短边 JPEG
results/<run_id>/           # 每次 run 一个目录
  result.json               # 全指标 + 耗时
  topk.npz                  # top-5 索引/概率/真值
src/  report/  video/  notebooks/  docs/
```
- `run_id` 命名:`<method>_<variant>_<seed>`,例 `resnet50_pretrained_42`、`bovw_sift512_42`。

## 2. 数据契约

### 2.1 `classes_500.json`(A 冻结,单一事实来源)
```json
{
  "seed": 42, "n_classes": 500,
  "classes": [
    {"class_id": 0, "category_id": 12873, "name": "Species name",
     "genus": "...", "family": "...", "kingdom": "..."}
  ]
}
```
- `class_id` 0–499 是**全项目唯一标签空间**,所有 `(N,500)` 概率矩阵按此列序。
- 带分类学层级(genus/family):Grad-CAM 分析"同属两个种"、误差分析按 genus/family 聚合都要用。

### 2.2 图像规格
- 统一 **256px 短边、JPEG** 存 `data/images/<class_id>/`。全队只做一次,打包传 Drive。
- **所有人必须用 `src/common.load_image(path) -> PIL.Image(RGB)`**,禁止自己 `cv2.imread`/`Image.open`
  (cv2 是 BGR、PIL 是 RGB,混用会让 E 的退化函数静默出错)。

### 2.3 manifest schema(`train.csv`/`val.csv`/`test.csv`)
```
path,class_id,split
data/images/0/abc.jpg,0,train
```
- 三份文件路径集合**两两交集为空**(泄漏检查脚本会断言);`test.csv` 全部来自官方 `val` split。

## 3. 方法接口(**整份契约的核心**)
所有方法(传统 / 从零 / 迁移)实现同一基类 `src/common.Method`:
```python
class Method:
    name: str
    def fit(self, train_df, val_df): ...          # 训练/建模(传统方法也走这里)
    def predict_proba(self, df, image_transform=None) -> np.ndarray:
        """返回 (len(df), 500) 概率矩阵,每行和为 1,列序 = class_id 0..499。
        image_transform: 可选 callable(PIL.Image)->PIL.Image,在 load_image 之后、
        送模型之前逐图施加。E 的鲁棒性扫描靠它,不碰任何人模型内部。"""
```
- ⚠️ **`image_transform` 参数从第一天就必须存在**(默认 None)。等模型冻结了再加=要回头改 B/C/D 三个人代码。
- E 跑鲁棒性:`method.predict_proba(test_df, image_transform=corruptions["gaussian_noise"](sev=0.1))`。

## 4. 结果契约
`src/common.save_result(run_id, y_true, probs, timings)` 写:
- `result.json`:`{run_id, method, top1, top5, overall_acc, balanced_acc, macro_p, macro_r, macro_f1, train_seconds, test_seconds, hardest_pairs:[...]}`
- `topk.npz`:`top5_idx (N,5)`、`top5_prob (N,5)`、`y_true (N,)`(约 160KB/run)。
- **不存完整 `(5000,500)` 概率矩阵**(10MB/run,撑爆仓库 + 顶到 ZIP≤40MB 上限);top-5 足以复算全部指标。

## 5. 评测契约
`src/evaluate.py` 只认 **manifest + `(N,500)` 概率矩阵**,对所有方法一视同仁:
- 输出:top-1 / top-5 / overall acc / balanced acc / macro-P/R/F1 / 混淆矩阵 / 耗时。
- 随机预测在 test 上应 top-1 ≈ 0.2%(1/500)、top-5 ≈ 1% —— **M1 验收就跑这个**。

## 6. 退化契约(E 提供,各方法 owner 自己跑)
`src/robustness.CORRUPTIONS: dict[str, (fn, [severities])]`,`fn(PIL, sev)->PIL`。
- ≥4 种:高斯噪声 / 高斯模糊 / 运动模糊 / 亮度对比度 / JPEG,每种 5 个严重度,参数化可复现。
- **只退化 test,不改训练、不重训**。各 owner 把 `fn(sev)` 当 `image_transform` 传进自己的 `predict_proba`。

## 7. Git 约定
- 分支:`main` 稳定;个人在 `feat/<role>-<topic>` 上开,PR 合入。
- **不入库**:`data/`、`*.pt`、`*.npz`、大图、`results/**/*.png`(gitignore 已配)。
- commit 讯息:`<role>: <what>`,例 `D: efficientnet_b0 transfer + randaugment ablation`。
- **代码 owner = A**(对应 Code 那 3 分),合并前过 `python -m py_compile src/*.py`。

## 8. 变更日志
- 2026-07-19 v1 初版(Molty 起草,对齐 PROJECT_PLAN)。kickoff 定稿后锁定。
