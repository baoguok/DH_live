"""
A helper function to get a default model for quick testing
"""
import os
import yaml
from omegaconf import open_dict, OmegaConf, DictConfig

import torch
from matanyone2.model.matanyone2 import MatAnyone2


def _resolve_interpolations(data: dict, root: dict) -> dict:
    """解析 YAML 中的 Hydra 风格插值，如 ${model.embed_dim}"""
    if isinstance(data, dict):
        return {k: _resolve_interpolations(v, root) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_interpolations(v, root) for v in data]
    if isinstance(data, str) and data.startswith("${") and data.endswith("}"):
        # 解析 ${a.b.c} 路径
        parts = data[2:-1].split(".")
        val = root
        for p in parts:
            val = val[p]
        return val
    return data


def _load_config() -> DictConfig:
    """手动加载合并配置，替代 Hydra 的 initialize/compose"""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

    # 读取主配置
    main_cfg_path = os.path.join(config_dir, "eval_matanyone_config.yaml")
    with open(main_cfg_path, "r", encoding="utf-8") as f:
        main = yaml.safe_load(f)

    # 读取 model/base.yaml（对应 defaults 中的 model: base）
    model_cfg_path = os.path.join(config_dir, "model", "base.yaml")
    with open(model_cfg_path, "r", encoding="utf-8") as f:
        model = yaml.safe_load(f)

    # 先合并 model 部分，供插值使用
    merged = dict(main)
    merged["model"] = model

    # 先解析 model 自身的插值（如 ${model.embed_dim}）
    merged["model"] = _resolve_interpolations(merged["model"], merged)
    # 再解析顶层的插值（如有）
    merged = _resolve_interpolations(merged, merged)

    # 删除 Hydra 私有字段，避免下游误用
    for key in ("defaults", "hydra"):
        merged.pop(key, None)

    return OmegaConf.create(merged)


def get_matanyone2_model(ckpt_path, device=None) -> MatAnyone2:
    cfg = _load_config()

    with open_dict(cfg):
        cfg["weights"] = ckpt_path

    # Load the network weights
    if device is not None:
        matanyone2 = MatAnyone2(cfg, single_object=True).to(device).eval()
        model_weights = torch.load(cfg.weights, map_location=device)
    else:  # if device is not specified, `.cuda()` by default
        matanyone2 = MatAnyone2(cfg, single_object=True).cuda().eval()
        model_weights = torch.load(cfg.weights)

    matanyone2.load_weights(model_weights)

    return matanyone2
