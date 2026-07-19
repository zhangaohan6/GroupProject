> ⚠️ **本文件已被 `docs/PROJECT_PLAN.md` 取代为主计划**(5 人分工 + 契约驱动)。
> 技术契约以 `docs/CONTRACTS.md` 为准;启动清单见 `docs/M0_KICKOFF.md`。
> 下面保留为快速索引。

# COMP9517 26T2 · Group Project — 作战计划(单一事实来源)

> 每次只更这一份。改了什么在底部「进度日志」记一行。

## 0. 硬信息
- **课程/权重**:COMP9517 Computer Vision · 占总分 **30%**(满分 30)
- **截止**:**2026-08-07(周五)22:00 AET** · 今天 2026-07-19,**约剩 3 周**(Weeks 6–10)
- **提交**:全组只由**一人**代交;video + report(PDF) + code(ZIP)
- **组队红线**:必须和 WebCMS 分配的组做,擅自换组 **−50%**。⚠️ 先确认自己在哪个组(z5633314)
- **学号**:z5633314

## 1. 任务
用 **iNaturalist-2021(iNat2021)mini 子集**做**细粒度物种分类**(植物/动物/真菌,很多物种长得像)。

## 2. 数据要求(评分实验最低配)
- **≥ 500 类**(从 10000 类里随机采 500,**记住 seed**)
- **每类 ≥ 50 张训练图** → 建议 **40 训 / 10 验**
- **每类 ≥ 10 张测试图** → 用**官方 val 集**当 test(官方 test 标签没公开)
- **严格三分、零泄漏**;**报告写清用了哪些类 + 每类张数 + seed**(可复现)
- 下载(train_mini 42GB / val 8.4GB,建议先只下我们采样的 500 类):
  - `train_mini.tar.gz` + `train_mini.json.tar.gz`
  - `val.tar.gz` + `val.json.tar.gz`
  - 官方仓:`github.com/visipedia/inat_comp/tree/master/2021`

## 3. 必做:≥ 2 个「真正不同」的完整方法
1. **传统管线**:手工特征 + 经典分类器 —— 例 **BoVW(SIFT)+ SVM**(或 HOG/LBP/颜色直方图 + SVM/RandomForest)
2. **深度管线**:CNN,**必须同时含**「**从零训练(随机初始化)**」和「**用预训练权重(ImageNet)微调**」两种

## 4. 指标(评分实验必算)
- **top-1、top-5 准确率**,overall accuracy,(可选 balanced acc)
- **macro 平均 Precision / Recall / F1**(细粒度重点看 **macro-F1**)
- **混淆矩阵**(全类可视化 + 挑最易混物种对细看)
- **训练/测试时间 vs 性能** 对比

## 5. 分数档位(照这个定目标)
- **23–26(comprehensive)**:方法有变化 + **系统消融**(一次只变一个设计)+ **训练曲线**(loss/acc)+ **误差分析**(混淆矩阵 + 最难物种对)
- **26–27**:深入做 **1 个** advanced 方向
- **28+**:深入做 **≥ 2 个** advanced 方向

### 🎯 推荐的 advanced 方向(冲 28+,打你的主场)
四选项:①Grad-CAM 解释性 ②测试期退化鲁棒性 ③类别数影响 ④持续学习。
**主推组合:② 鲁棒性 + ① Grad-CAM**,理由:
- **② 退化鲁棒性 = 你的评测/鲁棒性主场**:对 test 图加 ≥4 种退化(高斯噪声/高斯或运动模糊/亮度对比度/JPEG压缩),扫多个强度,画 **top-1 & macro-F1 vs 退化强度曲线**——**这就是你熟的 performance-vs-severity 曲线**。**模型不变、只退化 test**。
- **① Grad-CAM 便宜且互补**:可视化模型看的是物种本体还是背景;对/错样本、易混物种对的激活图对比,解释失败案例。
- 两个都**不用重训模型**,成本低、协同(② 量「掉多少」,① 解释「为什么」)→ 稳拿 28+。
- 次选叠加:③ 类别数影响(500→1000→2500,曲线),工程量中等。

## 6. 交付物 checklist
- **Video**:≤ 10 分钟 MP4(720p/1080p),≤ 100MB;**5 人全部出镜讲**(角落露脸+报名字,否则扣分);含**软件 demo**(可录屏剪辑)
- **Report**:**CVPR LaTeX 模板**(`\usepackage[pagenumbers]{cvpr}`),PDF,**正文 ≤ 10 页**(含图表,references 不计),≤ 10MB。**不用这个模板可能 0 分**
  - 结构:Introduction / Literature Review / Methods / Experimental Results / Discussion / Conclusion / References
- **Code**:ZIP ≤ 40MB,**只放自己的代码**(不含训练好的模型/图片/结果图),README 说明怎么跑 + 引用来源

## 7. 三周时间线(到 8/7)
- **第 1 周(7/19–7/25)· 打地基**
  - [ ] 确认 WebCMS 分组 + 建组内协作(GitHub repo / 共享盘)
  - [ ] 跑 `build_subset.py` 采样 500 类、40/10/10 三分、固定 seed
  - [ ] 传统管线跑通 baseline(BoVW+SIFT+SVM)
  - [ ] 深度管线跑通 baseline(ResNet 预训练微调,先出一条 acc 曲线)
- **第 2 周(7/26–8/1)· 主实验 + 消融**
  - [ ] 深度:从零 vs 预训练、换架构、数据增强 消融
  - [ ] 传统:换特征(SIFT/HOG/LBP)、换分类器(SVM/RF)消融
  - [ ] 全指标 + 混淆矩阵 + 训练曲线出齐
  - [ ] advanced ②:退化鲁棒性曲线
- **第 3 周(8/2–8/7)· advanced + 交付**
  - [ ] advanced ①:Grad-CAM 图 + 失败案例解释
  - [ ] 写 report(CVPR 模板)+ 录 video(5 人分工出镜)+ 整理 code ZIP
  - [ ] **8/6 预留缓冲**,8/7 前一人代交

## 8. 5 人分工(模板,按实到人填)
- **A(z5633314 / 你)**:深度管线 + advanced ②鲁棒性(你的主场)+ 评测/指标脚本
- **B**:传统管线(BoVW/SIFT/SVM)+ 特征消融
- **C**:深度架构消融 + 数据增强 + 训练曲线
- **D**:advanced ①Grad-CAM + 误差分析/混淆矩阵
- **E**:数据管线 + report 主笔 + video 统筹
- (每人都要碰**方法+代码+video+report**,不能只做一样——Week 11 有匿名互评,贡献太少会被调分)

## 9. 目录结构(契约 §5)
```
COMP9517-GroupProject/
├─ docs/  PROJECT_PLAN.md(总纲) · CONTRACTS.md(契约,以它为准) · M0_KICKOFF.md · SPEC-SUMMARY.md
├─ src/
│  ├─ common/  io.py(load_image ndarray, set_seed) · manifest.py(读取+泄漏校验)
│  ├─ data/    build_subset.py(采样+classes_500.json+manifests)          【A】
│  ├─ eval/    evaluate.py · analysis.py(混淆矩阵/易混对) · aggregate.py 【A】
│  ├─ methods/ base.py(Method基类+save_run) · deepnet.py(共享CNN)
│  │           traditional/【B】 scratch/【C】 transfer/【D】
│  └─ advanced/ robustness/(degradations+run)【E】 gradcam/【E】
├─ results/(json,入库) · predictions/(npz,入库) · report/ · video/ · notebooks/
└─ data/(不入库) images_256/ · manifests/ · classes_500.json · checkpoints/
```

## 10. 风险提醒
- **组队 −50% 红线**:先核对 WebCMS 分组
- **数据泄漏**:train/val/test 严格分开
- **收敛**:CNN 要训够(别欠拟合);存 checkpoint 可断点续
- **算力**:Colab 免费 T4 够(先只下采样的 500 类,几百 MB);数据放 Drive、每次 session 拷到本地盘再训
- **只能用 iNat2021 数据**做评分实验(自设 advanced 方向除外)

---
## 进度日志
- 2026-07-19 建目录 + 骨架代码 + 本计划(by Molty)。待:确认分组、下数据、跑两条 baseline。
- 2026-07-20 01:27 ⚠️ 图片首次抽取被父进程中断于 200 张;已改为完全脱离进程树重跑(见 logs/fetch.log)。计划/manifest 不受影响。
