# 团队契约 — COMP9517 小组项目

**状态**：草案，待 M0 kickoff 冻结
**冻结后的变更流程**：群里提出 → 说明影响面 → 全员确认 → A 更新本文件并标注版本 → 群里公告。**任何人不得单方面修改。**

---

## §1 总则

本文件规定 5 条工作线之间的所有接口。它存在的唯一目的是：**让 B、C、D、E 四个人能完全并行，且 E 的两项高级研究能在不碰任何人模型内部的前提下跑通所有方法。**

三条不可协商的规则：

1. **`test` 集只在最终评估时使用一次。** 调参、early stopping、模型选择一律只能用 `val`。
2. **`classes_500.json` 一旦冻结，`class_id` 映射永不重排。** 重排会让所有已跑出的结果全部作废。
3. **所有图像必须经由 `load_image()` 加载**（§3.3）。绕过它会让 `image_transform` 契约失效，鲁棒性研究直接报废。

---

## §2 数据契约

### 2.1 目录结构（在共享 Google Drive 上）

```
COMP9517_data/                     # 共享 Drive 根目录
  classes_500.json                 # 冻结的类别清单（同时提交进 Git）
  manifests/
    train.csv                      # 20,000 行 (500 类 × 40)
    val.csv                        #  5,000 行 (500 类 × 10)
    test.csv                       #  5,000 行 (500 类 × 10，来自官方 val split)
  images_256/
    <class_id>/                    # 000 ~ 499，三位零填充
      <original_filename>.jpg      # 短边缩放到 256px
  checkpoints/                     # 模型权重（不进 Git）
    <method>__<tag>/
```

Colab 工作流：**挂载 Drive → 把 `images_256/` 复制到 Colab 本地磁盘 → 再训练**。直接从 Drive 逐张读图会慢到无法接受。

### 2.2 `classes_500.json`

由 A 生成并冻结。`class_id` 是**局部**索引 0–499，与 iNat 官方的 `category_id`（0–9999）是两回事，必须双向记录。

```json
{
  "seed": 42,
  "n_classes": 500,
  "source_split": "train_mini",
  "created": "2026-07-21",
  "classes": [
    {
      "class_id": 0,
      "inat_category_id": 1234,
      "dir_name": "01234_Animalia_Chordata_Aves_Passeriformes_Corvidae_Corvus_corax",
      "kingdom": "Animalia",
      "phylum": "Chordata",
      "class": "Aves",
      "order": "Passeriformes",
      "family": "Corvidae",
      "genus": "Corvus",
      "species": "Corvus corax"
    }
  ]
}
```

> **为什么要把分类学层级拆出来**：E 做 Grad-CAM 时规范明确要求分析「易混淆物种对（例如**同属**的两个种）」，A 做误差分析时也需要按 genus/family 聚合来解释混淆模式。这些字段就编码在 iNat 的目录名里，采样时顺手解析出来，后面省掉大量返工。

### 2.3 Manifest CSV Schema

三份 manifest 列完全一致：

| 列名 | 类型 | 说明 |
|---|---|---|
| `filepath` | str | 相对 `images_256/` 的路径，如 `000/abc123.jpg` |
| `class_id` | int | 0–499 |
| `species` | str | 学名，仅供展示/画图 |

**约束**（A 在 M1 必须写断言脚本验证）：

- 三份 manifest 的 `filepath` 集合**两两交集为空**（无数据泄漏）
- `train` 每类恰好 40 行、`val` 每类恰好 10 行、`test` 每类恰好 10 行
- `test` 的每一行都来自官方 `val` split（**绝不能**来自 `train_mini`）
- 行顺序固定，提交进 Git 后不再改动

---

## §3 方法接口 ⭐ 最关键的契约

### 3.1 抽象基类

由 A 在 `src/methods/base.py` 提供，B/C/D 各自继承实现。

```python
from abc import ABC, abstractmethod
from typing import Callable, Optional
import numpy as np
import pandas as pd

ImageTransform = Callable[[np.ndarray], np.ndarray]


class Method(ABC):
    """所有方法（传统 / 从零 / 迁移）的统一接口。"""

    name: str      # 唯一标识，用于结果文件命名，如 "bovw_sift_svm"
    config: dict   # 全部超参数，会被原样写进结果 JSON（保证可复现）

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """用 train_df 拟合，用 val_df 做调参 / early stopping / 模型选择。
        绝对不允许在此接触 test 集。"""

    @abstractmethod
    def predict_proba(
        self,
        df: pd.DataFrame,
        image_transform: Optional[ImageTransform] = None,
    ) -> np.ndarray:
        """返回 shape (len(df), 500) 的 float32 概率矩阵。

        三条硬性约束：
          1. 行顺序必须与 df 的行顺序严格一致（否则所有指标静默出错）
          2. 列索引 j 对应 class_id == j（来自 classes_500.json）
          3. 每行经 softmax 归一化，行和为 1（top-5 指标依赖概率排序）

        image_transform: 可选。若提供，则在 load_image() 之后、
            本方法自身的预处理之前，对每张图施加。见 §3.2。
        """

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...
```

### 3.2 `image_transform` —— 为什么它必须从第一天就存在

E 的鲁棒性研究要求：**只退化测试图像，不改训练数据、不重训模型**，并且要在**所有方法**（含传统方法）上对比。

有了这个参数，E 的整个鲁棒性扫描就是：

```python
for deg_type, severity in sweep:
    probs = method.predict_proba(test_df, image_transform=degrade(deg_type, severity))
    metrics = evaluate(probs, test_df)
```

**E 完全不需要碰任何人的模型内部。** 反过来，如果第一天没定这个参数，等到 7/30 模型都冻结了再来加，意味着要回头改 B、C、D 三个人的代码——而那时只剩一周。

**语义规定**：`image_transform` 接收 `load_image()` 的输出（HWC、uint8、RGB），返回**同样格式**的数组。它施加在方法自身的 resize / 归一化 / 特征提取**之前**——模拟的是"相机拍到的就是一张糟糕的图"。

**不预生成退化图像**：5000 张 × 5 种退化 × 5 个等级 = 125,000 张、约 5GB。一律**在线施加**。

### 3.3 统一图像加载入口

由 A 在 `src/common/io.py` 提供。**所有人必须用它，不得自行 `cv2.imread` 或 `Image.open`。**

```python
def load_image(path: str) -> np.ndarray:
    """统一图像加载入口。返回 HWC、uint8、RGB 通道序的 ndarray。"""
```

> 理由：`cv2.imread` 返回 **BGR**，PIL 返回 **RGB**。如果 B 用 cv2、D 用 PIL，E 的同一个退化函数在两者上的行为就不一致（比如色彩相关的退化会左右颠倒），鲁棒性对比直接失去意义。

### 3.4 传统方法的概率输出

`LinearSVC` / `SGDClassifier` 没有 `predict_proba`。二选一：

- **推荐**：对 `decision_function` 的输出做 softmax，简单且足够（我们只需要排序正确）
- 或用 `CalibratedClassifierCV` 包一层（更规范但 500 类下训练开销明显更大）

选哪个由 B 决定，但**必须写进 `config` 并在报告里说明**。

---

## §4 结果契约

### 4.1 运行命名

```
<method>__<tag>[__<degtype>_s<severity>]__<split>
```

`tag` 是这次配置的短标签，如 `res50_ft_randaug`。示例：

- `transfer_resnet50__res50_ft_randaug__test`
- `transfer_resnet50__res50_ft_randaug__gaussian_noise_s3__test`

### 4.2 `results/<run>.json`（提交进 Git，很小）

```json
{
  "run": "transfer_resnet50__res50_ft_randaug__test",
  "method": "transfer_resnet50",
  "tag": "res50_ft_randaug",
  "split": "test",
  "degradation": null,
  "config": { "backbone": "resnet50", "lr": 0.001, "epochs": 30, "...": "..." },
  "metrics": {
    "top1": 0.0, "top5": 0.0,
    "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0,
    "balanced_accuracy": 0.0
  },
  "timing": { "train_seconds": 0.0, "predict_seconds": 0.0 },
  "n_samples": 5000, "n_classes": 500, "seed": 42,
  "git_commit": "abc1234",
  "timestamp": "2026-07-26T14:00:00+10:00"
}
```

退化运行把 `degradation` 填成 `{"type": "gaussian_noise", "severity": 3}`。

> **`timing` 不是可选项**——规范明确要求「compare the training and testing time (vs performance) of the different methods」。每次 run 都记，不要最后补测。

### 4.3 `predictions/<run>.npz`（提交进 Git）

**不要存完整的 (5000, 500) 概率矩阵**——每次 run 10MB，几十次 run 就撑爆仓库，而且规范要求提交的代码 ZIP ≤40MB。

只存 top-5 就足以复算全部要求的指标：

```python
np.savez_compressed(
    path,
    top5_idx=top5_idx,    # (N, 5) int16，按概率降序
    top5_prob=top5_prob,  # (N, 5) float32
    y_true=y_true,        # (N,)  int16
)
```

约 160KB/run。top-1、top-5、macro-P/R/F1、混淆矩阵、最易混淆物种对——**全部可以只从这三个数组算出来**。

完整概率矩阵只在需要算 mAP 之类的指标时保留，且**只放 Drive，不进 Git**。

---

## §5 仓库结构

```
comp9517-inat-project/
  README.md                  # 如何运行 + 外部库/借用代码来源（Code 3 分的评分点）
  requirements.txt
  .gitignore
  classes_500.json           # 冻结
  src/
    common/
      io.py                  # A: load_image(), set_seed()
      manifest.py            # A: manifest 读取与校验
    data/                    # A: 采样、流式抽取、manifest 生成
    eval/
      evaluate.py            # A: 全部指标
      analysis.py            # A: 混淆矩阵、最易混淆物种对
      aggregate.py           # A: 汇总所有 results/*.json 成报告表格
    methods/
      base.py                # A: Method 抽象基类
      traditional/           # B
      scratch/               # C
      transfer/              # D
    advanced/
      gradcam/               # E
      robustness/            # E
  notebooks/                 # Colab 入口 notebook，每人一个
  results/                   # JSON（提交）
  predictions/               # npz（提交）
  report/                    # LaTeX 源码（或用 Overleaf 另管）
```

---

## §6 Git 协作约定

### 6.1 分支策略

19 天、5 个人，**不搞 PR review 流程**（开销大于收益）。规则：

- 直接 push `main`
- push 前先 `git pull --rebase`
- 只改自己拥有的目录（见 §6.2）
- commit message 用中文或英文都行，但要说清楚改了什么

### 6.2 目录所有权 —— 把冲突降到接近零

| 目录 | owner | 其他人 |
|---|---|---|
| `src/common/`、`src/data/`、`src/eval/`、`src/methods/base.py` | **A** | 只读。要改就找 A |
| `src/methods/traditional/` | **B** | 只读 |
| `src/methods/scratch/` | **C** | 只读 |
| `src/methods/transfer/` | **D** | 只读 |
| `src/advanced/` | **E** | 只读 |
| `results/`、`predictions/` | 各自追加自己的文件 | 不删别人的 |
| `README.md`、`requirements.txt` | **A** 统稿 | 要加依赖先说一声 |

### 6.3 `.gitignore`

```gitignore
# 数据与权重一律不进 Git（同时满足提交 ZIP ≤40MB 且不含模型/图像的要求）
data/
images_256/
*.tar.gz
*.pth
*.pt
*.pkl
*.joblib
checkpoints/

# 完整概率矩阵（只留 predictions/*.npz）
*.npy

# 结果图像（规范明确要求 ZIP 里不要包含）
figures/
*.png
*.jpg

__pycache__/
.ipynb_checkpoints/
.DS_Store
```

> 这份 `.gitignore` 是照着**提交要求**写的：规范规定代码 ZIP ≤40MB 且「do not include trained models, input images, or result images」。从第一天就 ignore 掉，最后打包时直接 `git archive` 就合规，不用临时清理。

---

## §7 环境与可复现性

- **Python 版本**：以 Colab 当时的默认版本为准。A 在 M1 确认后写进 README，全员统一（本地开发的人用同一版本，避免 pickle 不兼容）。
- **依赖**：`requirements.txt` 由 A 统一维护并**钉住版本号**。
- **随机种子**：全局 `SEED = 42`。A 在 `src/common/io.py` 提供：

```python
def set_seed(seed: int = 42) -> None:
    """统一设置 random / numpy / torch（含 cuda）的随机种子。"""
```

每个训练脚本、每次采样开头都必须调用。

- **可复现性要求**：规范明确要求「clearly report exactly which classes you used and how many images per class」。`classes_500.json` + 冻结的 manifest + `config` 字段进结果 JSON，三者合起来满足这个要求。

---

## §8 检查清单（M1 验收，由 A 演示给全员）

- [ ] `classes_500.json` 已生成、已提交、已宣布冻结
- [ ] 三份 manifest 行数正确（20000 / 5000 / 5000）
- [ ] 泄漏断言脚本通过：三份 manifest 的 `filepath` 集合两两交集为空
- [ ] `test.csv` 全部来自官方 `val` split
- [ ] `images_256/` 已传到共享 Drive，其他 4 人都能挂载读到
- [ ] `load_image()` 在样例图上返回 HWC / uint8 / RGB
- [ ] **`evaluate.py` 用随机预测跑一遍 `test`，top-1 ≈ 0.2%（1/500）、top-5 ≈ 1%**

> 最后一条是整个评测框架的 sanity check。**数字对得上，说明指标实现和数据划分都没问题**；对不上就说明有 bug，必须当场解决，绝不能带着往下走。
