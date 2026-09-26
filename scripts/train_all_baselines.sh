#!/usr/bin/env bash
# 在 GPU 服务器上、datasets/ 已就绪后, 依次训练 5 个基线。
# 用法:
#   bash scripts/train_all_baselines.sh /path/to/datasets [/path/to/mit_b0.pth]
#
# 断点续跑 (中断后跳过已训完的, 从指定模型开始):
#   FROM=deeplabv3 bash scripts/train_all_baselines.sh datasets/ /root/weights/mit_b0.pth
#
# 显存适配 (默认按 ~11GB 显卡 + 512 随机裁剪标定; 验证集仍全图评测):
#   - 训练期随机裁剪 CROP=512, 大幅降低激活显存 (1024 全图训练在 11GB 卡上
#     UNet batch=16 必然 OOM);
#   - 各模型 batch 可用环境变量覆盖, 显存富余可调大, OOM 则调小, 如:
#       BATCH_UNET=16 bash scripts/train_all_baselines.sh datasets/ /root/weights/mit_b0.pth
#   - 大显存卡 (>=24GB) 想全图训练: CROP=1024 BATCH_UNET=4 BATCH_DEEPLAB=4 ...
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 缓解显存碎片 (CUDA OOM 官方建议)

DATASETS="${1:-datasets}"
MIT_CKPT="${2:-}"
CROP="${CROP:-512}"
BATCH_UNET="${BATCH_UNET:-8}"
BATCH_DEEPLAB="${BATCH_DEEPLAB:-4}"
BATCH_PSPNET="${BATCH_PSPNET:-4}"
BATCH_SEGFORMER="${BATCH_SEGFORMER:-8}"
BATCH_MASKFORMER="${BATCH_MASKFORMER:-4}"
COMMON=(--datasets "$DATASETS" --split-key split_a --epochs 80 --workers 4 --patience 20 --require-cuda --crop "$CROP")

# ---- 断点续跑: FROM=<模型名> 则跳过它之前的步骤 ----
_ALL="unet deeplabv3 pspnet segformer maskformer"
FROM="${FROM:-}"
if [[ -n "$FROM" && ! " $_ALL " == *" $FROM "* ]]; then
  echo "ERROR: FROM=$FROM 不是有效模型名 (候选: $_ALL)" >&2
  exit 1
fi
_REACHED=0
should_run() {
  if [[ -z "$FROM" || "$_REACHED" == 1 || "$1" == "$FROM" ]]; then
    [[ "$1" == "$FROM" ]] && _REACHED=1
    return 0
  fi
  echo "[skip] $1 (FROM=$FROM)"
  return 1
}

echo "配置: crop=$CROP | batch: unet=$BATCH_UNET deeplab=$BATCH_DEEPLAB psp=$BATCH_PSPNET segf=$BATCH_SEGFORMER maskf=$BATCH_MASKFORMER"
echo "HF_ENDPOINT=${HF_ENDPOINT:-<未设置>}  (timm 预训练骨干走 HF hub; 服务器连不上 huggingface.co 时先: export HF_ENDPOINT=https://hf-mirror.com)"

if should_run unet; then
echo "[1/5] UNet (from scratch)"
python -u scripts/train_baseline.py --model unet --pretrained 0 --batch "$BATCH_UNET" "${COMMON[@]}"
fi

if should_run deeplabv3; then
echo "[2/5] DeepLabV3 (torchvision COCO 预训练骨干+ASPP, 21→10 类头移植)"
python -u scripts/train_baseline.py --model deeplabv3 --pretrained 1 --batch "$BATCH_DEEPLAB" "${COMMON[@]}"
fi

if should_run pspnet; then
echo "[3/5] PSPNet (ResNet-50 ImageNet + PPM)"
python -u scripts/train_baseline.py --model pspnet --pretrained 1 --batch "$BATCH_PSPNET" "${COMMON[@]}"
fi

if should_run segformer; then
echo "[4/5] SegFormer (MiT-B0)"
if [[ -n "$MIT_CKPT" && -f "$MIT_CKPT" ]]; then
  python -u scripts/train_baseline.py --model segformer --backbone mit_b0 \
    --pretrained-from "$MIT_CKPT" --batch "$BATCH_SEGFORMER" "${COMMON[@]}"
else
  echo "WARN: 未提供 mit_b0.pth, SegFormer 将随机初始化。ImageNet 权重下载:"
  echo "  mkdir -p /root/weights && curl -L -o /root/weights/mit_b0.pth \\"
  echo "    https://huggingface.co/nvidia/mit-b0/resolve/main/pytorch_model.bin"
  echo "  (HF 官方 nvidia/mit-b0, 键名自动转换; NVlabs 原版 mit_b0.pth 亦可直载)"
  python -u scripts/train_baseline.py --model segformer --backbone mit_b0 --batch "$BATCH_SEGFORMER" "${COMMON[@]}"
fi
fi

if should_run maskformer; then
echo "[5/5] MaskFormer (ResNet50 ImageNet + Hungarian)"
python -u scripts/train_baseline.py --model maskformer --pretrained 1 --batch "$BATCH_MASKFORMER" "${COMMON[@]}"
fi

echo "全部基线训练结束。生成对比表:"
echo "  python scripts/compare_all.py --datasets $DATASETS"
