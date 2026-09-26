# DINO-Seg 泉州传统村落无人机高分遥感语义分割与要素占比评估

本项目针对泉州市沿海 12 个传统村落高分辨率无人机遥感正射影像数据集（1175 张切片，10 类地物），构建并训练基于 DINOv3 冻结骨干的语义分割模型 **DINO-Seg**，并自动化统计逐图及村级 10 类地物面积占比，支撑传统村落保护与空间演变评估。

---

## 一、系统架构与数据规格

### 1. 10 类地物体系

| ID | 英文名称 (`slug`) | 中文名称 | 色标 (RGB) | 大类归属 |
|---|---|---|---|---|
| 0 | `bare_soil` | 裸土 | [201, 162, 39] | 裸地类 (`g_bare`) |
| 1 | `cultivated_land` | 耕地 | [124, 179, 66] | 农用地 (`g_cultivated`) |
| 2 | `high_speed_rail_and_highway` | 高铁与高速公路 | [229, 57, 53] | 交通类 (`g_transport`) |
| 3 | `mountain_forest` | 山林 | [46, 125, 50] | 植被类 (`g_vegetation`) |
| 4 | `naked_mountain` | 荒山 | [141, 110, 99] | 裸地类 (`g_bare`) |
| 5 | `new_building` | 新建建筑 | [255, 152, 0] | 建筑类 (`g_building`) |
| 6 | `old_building` | 传统/老旧建筑 | [109, 76, 65] | 建筑类 (`g_building`) |
| 7 | `road` | 一般道路 | [158, 158, 158] | 交通类 (`g_transport`) |
| 8 | `tree` | 散生树木 | [27, 94, 32] | 植被类 (`g_vegetation`) |
| 9 | `water` | 水体 | [30, 136, 229] | 水体类 (`g_water`) |
| 255 | `ignore` | 图像填充/未标注区 | [0, 0, 0] | 排除统计 |

### 2. DINO-Seg 模型架构

- **Backbone**：Frozen DINOv3 ViT-B/16（`timm/vit_base_patch16_dinov3.lvd1689m`，86M 参数全部冻结，fp16 推理）；
- **Multi-Resolution Resample Blocks**：抽取 ViT 第 3、6、9、12 层的 patch tokens，重建成 1/16、1/8、1/4、1/2 四级特征金字塔；
- **Fusion Upsample Decoder**：跨层侧边连接 + 逐级融合上采样；
- **Segmentation Head**：轻量级 $1\times 1$ 卷积像素分类器（可训练参数共 7.56M）；
- **混合精度方案**：编码器前向 fp16，提取特征转为 fp32 进入解码器与梯度反传，杜绝 GradScaler 溢出 NaN。

### 3. 三档划分方案（应对空间自相关）

- **划分 A（Paper-comparable）**：按图随机划分（分层采样，70/15/15）；
- **划分 B（Village Holdout）**：留出村验证（彻底隔离地域特征）；
- **划分 C（Spatial Block）**：$1024\times 1024$ 空间聚类块划分，杜绝相邻切片重叠造成的空间泄漏。

---

## 二、评测结果与精度对标

在测试集（Split A，177 张独立验证瓦片）上的实测指标如下：

### 1. 逐类测试指标 (Test Split)

| 类别 | IoU | Precision | Recall | F1-Score |
|---|---|---|---|---|
| 水体 (`water`) | **0.9369** | 0.9416 | 0.9947 | 0.9674 |
| 山林 (`mountain_forest`) | **0.8461** | 0.9080 | 0.9255 | 0.9167 |
| 高铁/高速 (`high_speed_rail`) | **0.7296** | 0.9082 | 0.7876 | 0.8436 |
| 新建筑 (`new_building`) | **0.7103** | 0.8258 | 0.8354 | 0.8306 |
| 耕地 (`cultivated_land`) | **0.6908** | 0.7788 | 0.8601 | 0.8174 |
| 散树 (`tree`) | **0.4675** | 0.7153 | 0.5759 | 0.6380 |
| 一般道路 (`road`) | **0.4276** | 0.5284 | 0.6728 | 0.5920 |
| 荒山 (`naked_mountain`) | **0.3863** | 0.5342 | 0.5709 | 0.5519 |
| 老建筑 (`old_building`) | **0.3831** | 0.4414 | 0.7303 | 0.5503 |
| 裸土 (`bare_soil`) | **0.3624** | 0.5670 | 0.4640 | 0.5103 |

### 2. 总体指标与论文对标

| 指标 | 本地实测 (Macro) | 本地实测 (Micro / 全局) | 论文目标 | 说明 |
|---|---|---|---|---|
| **mIoU** | **0.5940** | **0.6677** (IoU_micro) | 0.7445 | 主导类优异，长尾小类（老建筑/裸土）拖累宏平均 |
| **Precision** | 0.7149 | **0.8007** | 0.8363 | 全局微平均精度已非常接近论文 (80.1% vs 83.6%) |
| **Recall** | 0.7417 | **0.8007** | 0.8702 | 召回率稳健 |
| **F1-Score** | 0.7255 | **0.8007** | 0.8522 | 全局 F1 达 0.80 |
| **Pixel Acc** | — | **0.8007** | — | 全图 80.1% 像素分类完全正确 |

### 3. 统计交付口径交叉验证（逐图占比 |GT - Pred| 偏差）

全量 1175 张遥感切片在**标注真实占比（GT）**与**模型预测占比（Pred）**对比下，每张切片的 10 类平均绝对误差：

$$\text{总平均偏差} = \mathbf{0.0249} \quad (2.49\%)$$

各类别单图平均偏差分布：
- 散生树木 (`p_tree`): 4.87%
- 山林 (`p_mountain_forest`): 4.49%
- 一般道路 (`p_road`): 3.15%
- 裸土 (`p_bare_soil`): 3.13%
- 荒山 (`p_naked_mountain`): 2.22%
- 耕地 (`p_cultivated_land`): 2.14%
- 新建建筑 (`p_new_building`): 1.55%
- 水体 (`p_water`): 1.52%
- 高铁公路 (`p_hsr`): 1.18%
- 老建筑 (`p_old_building`): **0.65%**

**结论**：尽管老建筑等小类单像素交并比（IoU）较低，但其面积总量的估计偏差仅 **0.65%**。模型输出的各类要素面积占比高度可信，完全满足村落空间要素演变与保护规划统计需求。

---

## 三、快速上手与使用指南

### 1. 环境准备

```bash
# 1. 克隆代码仓库
git clone https://github.com/javaver/123555.git
cd 123555

# 2. 安装基础依赖
pip install -r requirements.txt

# 3. 安装 PyTorch（以 CUDA 12.1 为例）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. 下载 DINOv3 预训练权重（国内镜像）
mkdir -p /root/weights
curl -L -o /root/weights/model.safetensors \
  https://hf-mirror.com/timm/vit_base_patch16_dinov3.lvd1689m/resolve/main/model.safetensors
```

### 2. 数据准备与预处理全流程

将原始数据（图片与 LabelMe JSON）放置于 `train/` 目录下：

```bash
# ① 数据入库与格式规范化（校验影像完整性、提取内嵌图、写出 manifest.csv）
python tools/ingest.py train/ --out datasets/

# ② 生成单通道分类掩膜（0-9 类 + 255 忽略区）
python tools/build_masks.py datasets/

# ③ 生成数据划分（同时生成划分 A、B、C）
python tools/split.py datasets/

# ④ 统计标注真实值（GT）的要素占比表与村级汇总
python tools/analyze.py datasets/ --mode gt
```

### 3. 模型训练

#### 方案一：端到端 GPU 训练（推荐，单卡 2080Ti 约 3 小时）
```bash
python -u scripts/train.py \
  --datasets datasets/ \
  --enc-ckpt /root/weights/model.safetensors \
  --batch 16 \
  --epochs 120 \
  --lr 3e-4 \
  --out runs/dinoseg/ > train.log 2>&1 &
```

#### 方案二：CPU / 慢机特征缓存极速训练
若本地无大显存 GPU，可先单次缓存冻结骨干特征，之后仅训练解码器（CPU 亦可在数分钟内跑完）：
```bash
# 一次性提取特征缓存（约 7.4GB）
python scripts/cache_features.py --datasets datasets/ --out runs/feats/ --enc-ckpt /root/weights/model.safetensors

# 仅训练解码器
python scripts/train.py --datasets datasets/ --cache-dir runs/feats/ --batch 32 --epochs 120 --out runs/dinoseg_fast/
```

### 4. 评估与指标输出

```bash
# 评估 Test 集（输出逐类与 Macro / Micro 表）
python scripts/eval.py --datasets datasets/ --ckpt runs/dinoseg/best.pt --split test

# 切换为空间去偏划分 C 评估
python scripts/eval.py --datasets datasets/ --ckpt runs/dinoseg/best.pt --split test --split-key split_c
```

### 5. 全量切片推理与村落占比统计

```bash
# ① 对全量 1175 张切片生成分割掩膜
python scripts/infer.py --datasets datasets/ --ckpt runs/dinoseg/best.pt --out runs/pred_masks/

# ② 依据预测掩膜统计逐图 10 类占比与 12 个村庄的汇总表
python tools/analyze.py datasets/ --mode pred --pred-dir runs/pred_masks/
```
生成的分析表包含：
- `datasets/analysis/proportions_pred.csv`：1175 行，每张图有效区域内 10 类地物独立占比及组合大类占比；
- `datasets/analysis/village_summary_pred.csv`：12 个村庄各自的各类面积加权均值汇总；
- 带 UTF-8-BOM 头，直接用 Excel 打开不乱码。

### 6. 任意尺寸新图推理（无人机正射大图）

对任意长宽尺寸的无人机航拍图进行自动化分割并输出占比：
```bash
python scripts/infer.py \
  --datasets datasets/ \
  --ckpt runs/dinoseg/best.pt \
  --image-dir new_image/ \
  --out runs/pred_new/
```
- 自动适配任意长宽，外扩补齐至 16 倍数并在推理完成后精确裁剪；
- 自动滤除非村庄区域（黑边填充）；
- 输出逐图单通道掩膜 PNG 及 `runs/pred_new/proportions.csv`。

### 7. 五大基线算法训练与论文双表（Table 1 + Table 2）对比生成

仓库内置了最终确定的 5 大基线算法（涵盖 CNN 跳跃连接、金字塔池化上下文聚合、空洞卷积金字塔、层级化自注意力与掩膜分类）：

```bash
# 训练任意基线 (服务器推荐 --pretrained 1 / SegFormer 用官方 mit_b0.pth)
python scripts/train_baseline.py --model unet --datasets datasets/ --pretrained 0 --batch 16 --require-cuda
python scripts/train_baseline.py --model deeplabv3 --datasets datasets/ --pretrained 1 --batch 8 --require-cuda
python scripts/train_baseline.py --model pspnet --datasets datasets/ --pretrained 1 --batch 8 --require-cuda
python scripts/train_baseline.py --model segformer --datasets datasets/ \
  --pretrained-from /root/weights/mit_b0.pth --batch 8 --require-cuda
python scripts/train_baseline.py --model maskformer --datasets datasets/ --pretrained 1 --batch 4 --require-cuda

# 或一键串行训练 5 个基线
bash scripts/train_all_baselines.sh datasets/ /root/weights/mit_b0.pth

# 一键生成论文双表 (缺权重默认报错; 仅预览论文对照时加 --allow-paper-fallback)
python scripts/compare_all.py --datasets datasets/
```

> **显存适配**：串训脚本默认 **训练期 512 随机裁剪 + 保守 batch**（按 ~11GB 显卡标定；1024 全图训练在 11GB 卡上 UNet batch=16 会 OOM；验证/测试仍全图评测，与 DINO-Seg 的 512 训练协议对齐）。显存富余可用环境变量调大：`BATCH_UNET=16 bash scripts/train_all_baselines.sh ...`；≥24GB 卡可 `CROP=1024` 全图训练（相应调小 batch）。
>
> **断点续跑**：串训中断后 `FROM=deeplabv3 bash scripts/train_all_baselines.sh ...` 从指定模型继续（跳过已训完的，如已完成的 UNet）。
>
> **多划分 (B/C)**：`SPLIT=split_b WITH_DINOSEG=1 bash scripts/train_all_baselines.sh ...` 自动写入 `runs_b/`（不覆盖 Split A 的 `runs/`）；出表 `python scripts/compare_all.py --datasets datasets/ --runs runs_b --split-key split_b`。第 6 步 DINO-Seg 用**原生全图协议**（120 轮，不裁剪）——Split A 实测 512 裁剪对齐版反而掉 ~4.5 点（val 0.536 vs ≥0.581，冻结 ViT 的全图全局上下文是其核心优势，见 `runs/dinoseg_aligned` 消融），故各模型用各自最优配方、同一数据划分/损失/类权重/评测口径对比。

- **DeepLabV3**：torchvision `deeplabv3_resnet50` **COCO 预训练**完整迁移（骨干+ASPP），再 21→10 类做 1×1 分类头移植（torchvision 硬校验不允许带 COCO 权重直接改类数）；权重文件 `deeplabv3_resnet50_coco-cd0a2569.pth` 在 `~/.cache/torch/hub/checkpoints/` 即离线加载；
- **PSPNet / MaskFormer 骨干**：timm ResNet-50 ImageNet 预训练（timm ≥0.9 经 HuggingFace hub 下载；服务器连不上 huggingface.co 时先 `export HF_ENDPOINT=https://hf-mirror.com`，下载失败会给出该提示后退出）；
- **SegFormer**：编码器为仓库内 **MiT（mit_b0..mit_b5）**，`--pretrained-from` 加载 ImageNet 权重——NVlabs 官方 `mit_bX.pth` 直载，或 HF 官方 [`nvidia/mit-b0`](https://huggingface.co/nvidia/mit-b0) 的 `pytorch_model.bin`（transformers 新旧键名布局自动转换，100% 覆盖率硬校验）；`--pretrained` 对 MiT 无效；
- **PSPNet**：ResNet-50（`output_stride=16` 空洞卷积，与 DeepLabV3 口径对齐）+ 金字塔池化模块（bins=1/2/3/6）聚合全局多尺度上下文；论文 4 要素对照值定稿后在 `scripts/compare_all.py` 的 `TODO(论文数值)` 处填入，此前无本地权重的 PSPNet 行以 `-` 占位；
- **MaskFormer**：匈牙利匹配损失（类别 CE，∅ 权重 0.1 + 掩膜 BCE + Dice）；
- 读取 `runs/<model>/best.pt`；**默认拒绝**缺权重时静默填论文数（避免把 `(paper)` 当成本地结果）；
- **Table 1** → `runs/comparison_table1.csv`；**Table 2** → `runs/comparison_table2.csv`。

---

## 四、主要文件清单

```
123555/
├── baselines/                         # 5 大基线算法独立模块
│   ├── unet/                          # 经典 U-Net (Skip Connections)
│   ├── deeplabv3/                     # DeepLabV3 (ResNet-50 + ASPP)
│   ├── pspnet/                        # PSPNet (ResNet-50 + 金字塔池化模块 PPM)
│   ├── segformer/                     # SegFormer (仓库内 MiT 编码器 + All-MLP Decoder)
│   │   └── mit.py                     # Mix Transformer (mit_b0..b5), 兼容官方权重键名
│   └── maskformer/                    # MaskFormer (Mask-Classification)
│       └── criterion.py               # 匈牙利二分图匹配损失 (CE + BCE + Dice)
├── configs/
│   └── dinoseg_vitb16_512.yaml        # 模型与训练超参数配置
├── docs/
│   └── PLAN.md                        # 项目实施全阶段详细规划方案
├── new_image/                         # 待推理的外部新无人机大图
├── runs/
│   ├── dinoseg/
│   │   ├── best.pt                    # 最佳检查点权重（val mIoU 0.5838）
│   │   └── metrics_test.csv           # 测试集指标表
│   ├── pred_masks/                    # 1175 张切片预测掩膜
│   ├── pred_new/                      # 新图预测掩膜与 proportions.csv
│   ├── comparison_table1.csv          # 论文 Table 1 四要素对比总表
│   └── comparison_table2.csv          # 论文 Table 2 十类逐类 IoU 对比总表
├── datasets/
│   ├── analysis/
│   │   ├── proportions_gt.csv         # 真实标注 1175 切片要素占比
│   │   ├── proportions_pred.csv       # 模型预测 1175 切片要素占比
│   │   ├── village_summary_gt.csv     # 12 村真实标注统计汇总
│   │   └── village_summary_pred.csv   # 12 村模型预测统计汇总
│   ├── manifest.csv                   # 全量影像清单
│   ├── masks/                         # 单通道 ground truth 掩膜
│   └── splits.csv                     # A/B/C 三档数据划分
├── scripts/
│   ├── train.py                       # DINO-Seg 训练主入口
│   ├── train_baseline.py              # 统一基线训练 (--pretrained / --pretrained-from / --require-cuda)
│   ├── train_all_baselines.sh         # 服务器一键串训 5 基线
│   ├── compare_all.py                 # 论文双表 (缺权重默认报错; --allow-paper-fallback 可选)
│   ├── eval.py                        # 评估入口（Macro + Micro 全口径）
│   ├── infer.py                       # 推理入口（支持全集切片与任意尺寸外部新图）
│   └── cache_features.py              # DINOv3 特征离线缓存
├── src/dinoseg/
│   ├── dataset.py                     # TileDataset 与 FeatureDataset
│   ├── losses.py                      # 类别加权交叉熵 + Dice 联合损失
│   ├── metrics.py                     # 混淆矩阵、逐类指标、Macro/Micro 统计
│   └── model.py                       # DINO-Seg 核心网络结构（fp16 混合精度）
└── tools/
    ├── ingest.py                      # 原始数据解析入库
    ├── build_masks.py                 # 多边形转掩膜
    ├── split.py                       # A/B/C 划分生成器
    ├── analyze.py                     # 要素占比统计与村级汇总分析
    ├── tiling_check.py                # 瓦片空间重叠度核查
    └── debug_nan.py                   # 梯度与特征数值稳定性探测
```

---

## 五、复现与升级说明

1. **数值稳定性保证**：ViT 高层特征模长通常在 1000 以上，在纯 fp16 反向传播中极易造成 GradScaler 上溢 NaN。本项目采用解码器强制 fp32 模式，确保百轮以上长时间训练 100% 稳定收敛。
2. **更高精度进阶选项**：
   - **TTA 测试时增强**：推理时结合水平/垂直翻转平均，通常可直接带来 1–2% mIoU 提升；
   - **升级骨干网络**：如需进一步迫近 0.7445 指标，可将 backbone 替换为针对遥感预训练的 `timm/vit_large_patch16_dinov3.sat493m`。
