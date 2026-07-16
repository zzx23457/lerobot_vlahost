#!/usr/bin/env python3
"""中文优化版启动脚本 - 带文件浏览器

包含所有修复和文件浏览器功能
"""

import sys
import warnings
from pathlib import Path

# 过滤所有警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="subprocess")

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from workflows.robot_interaction.ui.app_zh import main_zh

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 启动 LeRobot 统一控制界面（增强版）")
    print("=" * 60)
    print("✨ 新功能:")
    print("   - 文件浏览器（快速选择模型和数据集）")
    print("   - 相机预览完全修复")
    print("   - 所有警告已消除")
    print("=" * 60)
    print("📍 访问地址: http://localhost:7860")
    print("🌐 局域网访问: http://<本机IP>:7860")
    print("🛑 停止: 按 Ctrl+C")
    print("=" * 60)
    print()
    main_zh()
