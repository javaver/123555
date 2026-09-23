# DINO-Seg 传统村落景观语义分割 · 实施方案

> 版本：v1（2026-09-23）　|　分支：`arena/01a0cdbe-123555`
> 目标：在泉州沿海 12 个传统村落无人机数据集（10 类）上实现并验证 DINO-Seg
> （冻结 DINOv3 编码器 + 多尺度重采样块 + 融合上采样解码器 + 轻量分割头），
> 复现/对标论文指标 **Precision 0.8363 / Recall 0.8702 / F1 0.8522 / IoU 0.7445**，
> 并输出**每张图片 10 类要素（bare soil … water）的占比统计**（可聚合到村），
> 作为传统村落保护评估的量化依据。

---

## 0. 一页摘要

| 阶段 | 内容 | 产出 | 估时 |
|---|---|---|---|
| M1 数据工程 | 格式修复、LabelMe→栅格掩膜、ignore 掩膜、划分、体检报告 | `datasets/masks/`、`manifest.csv`、`docs/DATA_AUDIT.md` | 1–2 天 |
| M2 基线 | UNet / DeepLabV3+ / SegFormer-B2 同口径训练评估 | 基线指标表 v1 | 2–4 天 |
| M3 DINO-Seg | 冻结 DINOv3-ViT-B/16（卫星预训练优先）+ 重采样块 + 融合解码器 + 头 | 主模型 ckpt、指标表（对标论文） | 3–5 天 |
| M4 消融与分析 | 编码器/解码器消融；逐图 10 类要素占比统计 | 消融表、逐图占比 CSV、占比条形图 | 2–3 天 |
| M5 收尾 | README、复现脚本、报告 | 发布包 | 1–2 天 |

硬件口径：单卡 24 GB（fp16/AMP）可跑 ViT-B 冻结方案；训练在用户机器/云执行，**本沙箱无 GPU 且未装 torch，仅做数据核验与文档**（见 §10）。

---

## 1. 数据现状审计（已用工具实测，均可复跑）

仓库现有 `train/` 下 **9 组样本**（图 + LabelMe JSON）与 `tools/dataset_check.py`。
以下结论全部由本仓库脚本实测得出（复跑命令见各条括号）：

1. **容器格式**：9 个 `*.png` 的魔数均为 `49 49 2a 00`（小端 **TIFF**，未压缩，RGB 8-bit，
   `ImageDescription={"shape": [512, 512, 3]}`；PIL 按内容识别可正常读取，
   但按扩展名分派解码器的加载器（如 OpenCV `imread`）会解码失败）。
   （`python3 tools/dataset_check.py train/` → "真实容器格式: {'tiff': 9}"）
2. **图可自 JSON 恢复**：JSON 内嵌 `imageData`（base64 真 PNG）与磁盘文件**逐像素相同**
   （9/9，`mean|diff| = 0`）。⇒ 全量数据若缺原图，可用内嵌图兜底；也意味着存储冗余，
   入仓只保留 JSON 即可还原原图。
3. **尺寸一致**：文件/JSON 头/内嵌图三方均为 512×512，0 组不一致。
4. **标注**：LabelMe v5.6.0 与 v5.10.1 混存（无碍）；`shape_type` 全为 `polygon`；
   坐标范围 [−1e-15, 511]（栅格化时 clip 到 [0,512)）。
5. **村落与命名**：`imagePath` 为 Windows 反斜杠相对路径 `..\{village}14\tile_{x}_{y}.png`；
   已见 4 村：`sanglincun14、xushancun14、shuangxicun14、fengmeicun14`（12 村中的 4 村）；
   平铺目录下同名瓦片用 `_1、_2` 后缀区分 ⇒ **样本主键必须用 (village, tile)，不能用文件名**。
6. **无效区（padding）**：标注多边形并集之外为近黑填充（max 通道 p99 = 3，而域内 p1 ≥ 8）。
   规则 `max(R,G,B) ≤ 5` 与"多边形并集补集"的一致性在 9 张上为 **99.42%–99.99%** ⇒
   ignore 掩膜 = 多边形并集补集 ∪ (max 通道 ≤ 5)，双保险。
7. **类别覆盖（样本）**：仅出现 4/10 类（mountain forest、naked mountain、road、new building）；
   其余 6 类（bare soil、cultivated land、high speed rail and highway、old building、tree、water）
   应在未上传的其余 1166 组中出现——M1 体检须对全量出"类别×像素"直方图并预警稀有类。
8. **多边形重叠**：存在但极小（边界细条，1–322 px）⇒ 栅格化按**面积升序**绘制（小目标后画、覆盖大目标边缘），避免大路/大楼被山林吞边。

类别 ID 约定（与 `tools/dataset_check.py` 一致）：

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | ignore |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 类 | bare soil | cultivated land | high speed rail & highway | mountain forest | naked mountain | new building | old building | road | tree | water | 255 |

---

## 2. 总体流程

```
raw(上传的 2350 文件)
  │  tools/ingest.py        —— 格式修复(TIFF→真PNG)、主键(village,tile)、内嵌图兜底
  ▼
datasets/images/{village}/{tile}.png
  │  tools/build_masks.py   —— 多边形面积升序栅格化 + ignore 掩膜(§1.6) + 一致性体检
  ▼
datasets/masks/{village}_{tile}.png (0..9, 255=ignore)
  │  tools/split.py         —— 划分 A: 按瓦片分层 70/15/15(对标论文口径)
  │                           划分 B: 按村庄留出 9训/1验/2试(泛化口径)
  ▼
训练 ──► 评估(P/R/F1/mIoU, 宏+微) ──► tools/analyze.py 逐图10类占比统计 ──► CSV/图表
```

全部大文件（images/masks/runs/ckpt）**不进 git**，用 `.gitignore` 排除；
进 git 的只有代码、配置、`manifest.csv`（每瓦片一行的小表）与文档。

---

## 3. 数据工程（M1）

### 3.1 `tools/ingest.py`
- 扫描任意目录布局；从 JSON `imagePath` 正则提取 `(village, tile)`，统一 `/` 分隔；
- 容器修复：`sniff()`（复用 dataset_check 的魔数识别）→ 用 PIL 重存为真 PNG（统一下游）；
  磁盘缺失时解码 `imageData` 兜底，并打 `recovered=1` 标记；
- 输出 `datasets/manifest.csv`：`village,tile,image,mask,ignore_pct,per_class_px...,recovered,source`。

### 3.2 `tools/build_masks.py`
- 多边形 → 类别掩膜：PIL `ImageDraw.polygon`，**按多边形面积升序**绘制（§1.8）；
- ignore = `(mask==0 且不在任一并集内)` 即并集补集，再 ∪ `(max 通道 ≤ 5)`；
- 体检断言：三方尺寸一致、标签 ∈ 10 类、坐标在界、overlap 比例 < 1%（超阈值报警人工复核）。

### 3.3 划分
- **划分 A（主，对标论文）**：按瓦片随机 70/15/15，种子 42；分层键 = 出现的类别集合，
  保证稀有类（old building / water / high speed rail & highway）三折都出现；
- **划分 B（辅，泛化）**：村庄级留出 9/1/2，报告"未见村庄"性能，支撑保护评估外推性论述；
- 两划分互不污染：B 的测试村庄在 A 中完全不参与任何统计（类别权重只从训练集算）。

### 3.4 增强与归一化
- 几何：随机水平/垂直翻转、90° 倍数旋转（正射影像无方向先验）、RandomScaleCrop(0.5–1.5)→512；
- 光度：ColorJitter(0.2/0.2/0.2/0.05)；**统计量只在非 ignore 像素上计算**（避免近黑 padding 拉偏）；
- 归一化：ImageNet mean/std 起步，M3 做"域内均值方差 vs ImageNet"消融；
- 稀有类：按训练集像素频率 `w = median(freq)/freq` 截断到 [1, 20] 作 CE 权重；备选 Focal(γ=2)。

---

## 4. 模型设计：DINO-Seg（M3）

输入 512×512×3。所有模块参数见下表（ViT-B 口径，可训练部分 ~18M）。

```
 image 512² ──► [Frozen DINOv3 ViT-B/16] ──► 最后4个block的 patch tokens
   每块 32×32×768（ViT-B 共12层，取第9–12层），丢弃 CLS 与 4 个 register tokens
        │
        ▼  Resample Blocks ×4（重建多分辨率特征）
   每级: Linear768→C_i + 3×3conv → learned upsample(pixel-shuffle) 到 {32,64,128,256}²
        C = {768,384,192,96}→统一 proj 到 256
        ▼  Fusion Upsample Decoder（DPT 式自顶向下）
   F4=256²? 逐级: up(×2, bilinear+conv) + conv3×3(fuse) + residual → 128²×256 → 256²×128
        ▼  Lightweight Seg Head
   conv3×3(128→128) + BN + ReLU + conv1×1 → 10 logits @256² ──bilinear──► 512²
```

- **编码器**：冻结，`facebook/dinov3-vitb16-pretrain-sat493m`（**卫星域预训练**，与航拍同域，优先）
  否则 `-pretrain-lvd1689m`；ViT-L/16 作上限消融。注意 DINOv3 有 4 个 register tokens，
  取特征时须剔除（CLS+registers），否则网格错位。
- **Resample block**：token→2D 后用 PixelShuffle(2) 学习式上采样（对比 bilinear 作消融），
  每级一个 3×3 conv 精修；作用 = 把单尺度 ViT 特征"重建"成 FPN 式多分辨率。
- **融合上采样解码器**：自顶向下 ×2 上采样 + 卷积融合 + 残差（DPT/SegFormer 混合式），
  输出 1/4 分辨率（256²），头内再上采样或损失在 256² 上算（GT 下采样，**近邻**插值）。
- **头**：2 层 conv，10 类；可训练参数 ~18M（ViT-B 86M 冻结不计梯度）。
- **损失**：`CE(ignore_index=255, class_weight) + Dice(逐类, ignore 外)`，1:1；
  可选边界辅助头（Sobel 边界 BCE）放 M4 消融。
- **训练**：AdamW lr 8e-4（可训练部分）/ cos 退火到 1e-6，warmup 5%，
  batch 8×512²·卡（AMP fp16，grad-clip 1.0），epoch 100–150，early-stop on val mIoU（patience 15），
  EMA 0.999 作评估权重；seed 42/0/1 三次报均值±std。
- **基线**：UNet(R50)、DeepLabV3+(R50)、SegFormer-B2、Mask2Former(Swin-T, 可选)；
  同划分 A、同增强、同 seed 口径；另加 **DINOv2-ViT-B 同架构消融**，隔离"DINOv3 的贡献"。

---

## 5. 评估协议（M2/M3）

- 指标：每类 Precision/Recall/F1/IoU + **宏平均**（论文口径待确认，默认宏；同时报微平均/像素 Acc）；
  ignore(255) 像素全程剔除。
- 对标：划分 A 测试集上对比论文 0.8363/0.8702/0.8522/0.7445；若口径不一致（如论文为微平均），
  在表中两列并列并注明。
- 稀有类单独成表（old building / water / high speed rail & highway 的 P/R/F1）；
- 混淆矩阵 + 按村庄子集结果（划分 B）；3 seed 均值±std；checkpoint 按 val mIoU 选。

---

## 6. 逐图要素占比统计（M4，核心交付）

`tools/analyze.py` 对**每张图片**计算 10 类要素占比，一张图一行：

1. **掩膜来源两种模式**：`gt`（直接用标注掩膜，给数据集做要素画像）与
   `pred`（模型推理输出，对全部 1175 张出占比）；
2. **分母 = 有效像素**（剔除 ignore=255 的近黑 padding；实测单张 padding 最高约 90%，§1.6，
   用全图 512² 当分母会把占比严重稀释）。同时输出 `valid_pct` 列记录有效区占全图比例；
3. **CSV 列**：`village, tile, source, valid_pct, p_bare_soil, p_cultivated_land,
   p_hsr_highway, p_mountain_forest, p_naked_mountain, p_new_building,
   p_old_building, p_road, p_tree, p_water`（10 个占比在有效区内和为 1）；
4. **组合占比**（配置里定义映射，随表附列）：建筑 = new building + old building；
   植被 = tree + mountain forest；水体 = water；交通 = road + hsr & highway；
   裸露地 = bare soil + naked mountain；耕地 = cultivated land；
5. **聚合（可选）**：按村 / 全数据集以有效像素加权平均得汇总表；
   逐图占比堆叠条形图、村级对比图，供论文图件。

---

## 7. 仓库结构（落地后）

```
tools/            dataset_check.py  ingest.py  build_masks.py  split.py  visualize.py  analyze.py
src/dinoseg/      model.py  dataset.py  losses.py  metrics.py
configs/          dinoseg_vitb16_sat_512.yaml  baselines/{unet,dlv3p,segformer}.yaml
scripts/          train.py  eval.py  infer.py
docs/             PLAN.md  DATA_AUDIT.md(全量体检后生成)
datasets/, runs/  → .gitignore（大文件外置）
```

工程约定：yaml 固化全部超参；manifest/指标 CSV 入 git；ckpt 与图像不入 git；
`requirements.txt`（torch≥2.1、transformers 或 timm≥1.0.20 含 DINOv3 权重映射、Pillow、numpy、scipy、matplotlib）。

---

## 8. 风险与开放问题（需用户确认）

1. **全量标注格式**：方案假定与样本一致（LabelMe JSON + 内嵌图）。若部分村庄交付的是
   彩色掩膜 PNG，`build_masks.py` 需加 palette 映射分支——请确认。
2. **论文划分与口径**：复现精确数字需要同 train/val/test 划分及宏/微口径；若不可得，
   以本文 §3.3 划分 A 为准并显著注明（数字不具逐位可比性）。
3. **算力**：本沙箱无 GPU/torch；训练需在用户侧执行，M2/M3 估时按单卡 24GB 计。
4. 村庄名后缀 `14` 是否为影像年份（2014）？若存在多期影像，§6 的逐图占比可直接扩展成时序对比。
5. GSD / 正射分辨率为可选项：默认交付就是像素占比；若提供 GSD，CSV 可加挂 m² 面积列。
6. DINOv3 许可：DINOv3 License 允许商用（含署名条款），学术使用无碍；交付物中附许可证说明。

---

## 9. 本阶段（M0）已完成

- 数据审计（§1 全 8 条，脚本实测）；`.scratch/` 下留有两版九宫格目检图
  （`contact_sheet.png`、`contact_bright.png`，已被 gitignore）；
- 确认 DINOv3 公开族系与 HF 检查点命名（含卫星预训练 sat493m），timm≥1.0.20 已支持权重映射；
- 本方案文档。

---

## 10. 实施进展

- **M0**：数据审计（§1 全 8 条）+ 本方案文档。
- **M1（已实现，仓库 9 组样本验收通过）**：
  - `tools/ds_common.py`：类别表、魔数识别、LabelMe 解析、manifest 读写；
  - `tools/ingest.py`：入库 9 组 / 4 村，0 告警，TIFF→真 PNG，内嵌图兜底；
  - `tools/build_masks.py`：9 张掩膜，取值集合 {3,4,5,7,255} ⊆ {0..9}∪{255}，回写像素统计；
  - `tools/split.py`：划分 A/B 写 `splits.csv`（小样本分层告警属预期，全量数据正常）；
  - `tools/analyze.py`：gt/pred 双模式逐图占比 + 村级加权汇总；两模式互查 max|Δp| ≤ 0.011
    （差异仅来自分母口径：多边形并集 vs padding 规则）。
- **本地全量运行**（仓库根目录，依赖仅 Pillow+numpy）：

  ```bash
  python tools/ingest.py train/ --out datasets/
  python tools/build_masks.py datasets/
  python tools/split.py datasets/
  python tools/analyze.py datasets/ --mode gt
  # 模型训完后对全部图片出占比:
  python tools/analyze.py datasets/ --mode pred --pred-dir runs/pred_masks/
  ```
- **M2/M3（已实现，CPU 冒烟验证通过）**：
  - `src/dinoseg/model.py`：FrozenDINOv3Encoder（timm `vit_base_patch16_dinov3.lvd1689m`，
    最后 4 block patch tokens，剔 CLS+registers）+ ResampleBlock（亚像素上采样）
    + FusionUpsampleDecoder + 轻量头；可训练 7.56M / 总 93.2M；
  - `src/dinoseg/{dataset,losses,metrics}.py`、`scripts/{train,eval,infer}.py`、
    `configs/dinoseg_vitb16_512.yaml`；
  - 冒烟（随机初始化、9 样本、CPU）：train 1 epoch、eval 出论文口径表、
    infer 9 张掩膜、analyze pred 模式全通；
  - 预训练权重下载未在沙箱验证（HF 被墙）：模型 ID 已被 timm 识别（进入 dinov3 factory），
    用户侧若下载受阻设 `HF_ENDPOINT=https://hf-mirror.com`。
- **空间泄漏对策（对用户质疑的回应）**：划分 A 为按图随机(仅主导类分层)，
  不抗空间自相关；新增 `tools/tiling_check.py`（全量验证切瓦步长：512=无重叠网格，
  <512=重叠切分需警惕真重复泄漏）与**划分 C 空间块**（1024² 成块整块进折，
  splits.csv 增 split_c 列；train/eval/infer 增 `--split-key split_c` 可切换）。
  论文三档并报：A 对标论文、C 空间去偏、B 整村留出。
