#!/usr/bin/env bash
# 在 GPU 服务器上、datasets/ 已就绪后, 依次训练 5 个基线。
# 用法:
#   bash scripts/train_all_baselines.sh /path/to/datasets [/path/to/mit_b0.pth]
set -euo pipefail
DATASETS="${1:-datasets}"
MIT_CKPT="${2:-}"
COMMON=(--datasets "$DATASETS" --split-key split_a --epochs 80 --workers 4 --patience 20 --require-cuda)

echo "[1/5] UNet (from scratch)"
python -u scripts/train_baseline.py --model unet --pretrained 0 --batch 16 "${COMMON[@]}"

echo "[2/5] DeepLabV3 (ImageNet)"
python -u scripts/train_baseline.py --model deeplabv3 --pretrained 1 --batch 8 "${COMMON[@]}"

echo "[3/5] DPT (ImageNet ViT)"
python -u scripts/train_baseline.py --model dpt --pretrained 1 --batch 4 "${COMMON[@]}"

echo "[4/5] SegFormer (MiT-B0)"
if [[ -n "$MIT_CKPT" && -f "$MIT_CKPT" ]]; then
  python -u scripts/train_baseline.py --model segformer --backbone mit_b0 \
    --pretrained-from "$MIT_CKPT" --batch 8 "${COMMON[@]}"
else
  echo "WARN: 未提供 mit_b0.pth, SegFormer 将随机初始化。建议:"
  echo "  curl -L -o /root/weights/mit_b0.pth https://github.com/NVlabs/SegFormer/releases/download/v1.0/mit_b0.pth"
  python -u scripts/train_baseline.py --model segformer --backbone mit_b0 --batch 8 "${COMMON[@]}"
fi

echo "[5/5] MaskFormer (ResNet50 ImageNet + Hungarian)"
python -u scripts/train_baseline.py --model maskformer --pretrained 1 --batch 4 "${COMMON[@]}"

echo "全部基线训练结束。生成对比表:"
echo "  python scripts/compare_all.py --datasets $DATASETS"
