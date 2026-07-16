"""workflows/_config_loader.py — workflows/*.py 共享的配置加载工具

支持 .json / .yaml / .yml，按后缀自动分发到对应解析器。

优先级（用户传入的路径后缀决定格式）：
    .yaml / .yml → yaml.safe_load
    .json        → json.load
    其他         → 尝试按 YAML 解析（更宽容），失败再回落到 JSON

向后兼容：
    - 旧 deploy_config.json / replay_config.json 文件继续可用（_comments 字段
      会被自动剔除，不会污染下游）。
    - 调用方代码完全无感：load_config(path) 返回的 dict 结构与之前一致。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(config_path: Path) -> dict[str, Any]:
    """加载配置文件，按后缀选择解析器。

    Args:
        config_path: 配置文件路径。必须存在。

    Returns:
        解析后的 dict。

    Raises:
        SystemExit: 文件不存在时。
        ValueError: 文件内容不是合法配置格式时。
    """
    if not config_path.is_file():
        raise SystemExit(f"❌ 配置文件不存在: {config_path}")

    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        data = _load_yaml(text, source=config_path)
    elif suffix == ".json":
        data = _load_json(text, source=config_path)
    else:
        # 未知后缀：先试 YAML（更宽容，注释也能解析），失败回落 JSON
        try:
            data = _load_yaml(text, source=config_path)
        except Exception:
            data = _load_json(text, source=config_path)

    # 兼容旧版 _comments 字段（deploy_config.json / replay_config.json 用过）
    if isinstance(data, dict):
        data.pop("_comments", None)

    return data


def _load_yaml(text: str, source: Path) -> dict[str, Any]:
    import yaml  # 局部 import，避免冷启动开销

    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"YAML 解析失败 ({source}): {e}") from e
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML 顶层必须是 mapping（{source}），实际是 {type(loaded).__name__}")
    return loaded


def _load_json(text: str, source: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败 ({source}): {e}") from e
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON 顶层必须是 object（{source}），实际是 {type(loaded).__name__}")
    return loaded
