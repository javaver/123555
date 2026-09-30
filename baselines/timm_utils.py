"""timm 骨干构建公共工具。

timm >= 0.9 的预训练权重统一从 HuggingFace hub 拉取; 服务器若无法直连
huggingface.co, 会在 create_model 时抛网络异常。这里统一包装, 把失败转成
可操作的中文提示 (设 HF 镜像 / 离线放置缓存), 避免长训练脚本中途崩在
晦涩的 traceback 上。
"""
from __future__ import annotations


def create_timm_encoder(backbone: str, pretrained: bool, **kwargs):
    """timm features_only 骨干; 下载失败时给出镜像/缓存的可操作提示。"""
    import timm

    try:
        return timm.create_model(backbone, features_only=True, pretrained=pretrained, **kwargs)
    except Exception as e:
        if not pretrained:
            raise
        raise RuntimeError(
            f"timm 预训练骨干下载失败 ({backbone}): {e}\n"
            "服务器网络不通 huggingface.co 时, 先设镜像再重跑:\n"
            "  export HF_ENDPOINT=https://hf-mirror.com   (建议写入 ~/.bashrc)\n"
            "若镜像也不可达, 在有网的机器上下载对应权重后放入 ~/.cache/huggingface/, "
            "或临时改用 --pretrained 0 随机初始化 (会拉低该基线上限, 仅救急用)。"
        ) from e
