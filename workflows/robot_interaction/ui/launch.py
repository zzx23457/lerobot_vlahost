#!/usr/bin/env python3
"""中文优化版启动脚本 - yaml-centric

包含所有修复、文件浏览器功能和 proxy 兼容性处理
"""

import os
import sys
import warnings
from pathlib import Path

# gradio 通过 httpx 拉资源时不支持 socks:// 代理 scheme；如果用户的 shell
# 里有 ALL_PROXY=socks://... 或类似的代理设置，会在 gradio import 时
# 直接 ValueError。本 UI 只服务本地 0.0.0.0:7860，不需要任何代理，
# 所以在 import gradio 之前先把环境里的代理变量清掉。
for _proxy_var in (
    "ALL_PROXY", "all_proxy",
    "HTTPS_PROXY", "https_proxy",
    "HTTP_PROXY", "http_proxy",
):
    os.environ.pop(_proxy_var, None)

# 过滤所有警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="subprocess")

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from workflows.robot_interaction.ui.app_zh import main_zh  # noqa: E402

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 启动 LeRobot 统一控制界面（yaml-centric）")
    print("=" * 60)
    print("✨ 特性:")
    print("   - YAML 直接编辑（与 deploy_config*.yaml 格式一致）")
    print("   - 表单视图作为可选的结构化编辑入口")
    print("   - 模板加载：deploy_config_chunk / deploy_config_hybrid / replay_config")
    print("   - 启动 = 写 yaml 到 tempfile + 脚本 --config <tmp>")
    print("   - 已自动清除 ALL_PROXY 等 socks 代理（gradio 兼容性）")
    print("=" * 60)
    print("📍 访问地址: http://localhost:7860")
    print("🌐 局域网访问: http://<本机IP>:7860")
    print("🛑 停止: 按 Ctrl+C")
    print("=" * 60)
    print()
    main_zh()