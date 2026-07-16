#!/usr/bin/env python3
"""测试 HTTP 机器人接口的基本功能

运行前确保：
1. HTTP 服务器已启动在 http://192.168.10.123:8010
2. 已安装项目依赖（uv sync 或 pip install -e .）

用法:
    python workflows/robot_interaction/test_http_robot.py
"""
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from lerobot.robots.marvain_m6_http import MarvainM6HttpRobotConfig, MarvainM6HttpRobot
    print("✓ 成功导入 MarvainM6HttpRobot 和 MarvainM6HttpRobotConfig")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保已安装项目依赖")
    sys.exit(1)

# 测试配置创建
try:
    config = MarvainM6HttpRobotConfig(
        id="test_robot",
        http_base_url="http://192.168.15.123:8010",
        timeout=5.0,
        cameras={},
        joint_names=[f"joint_{i}" for i in range(16)],
    )
    print(f"✓ 配置创建成功: robot_id={config.id}, url={config.http_base_url}")
except Exception as e:
    print(f"❌ 配置创建失败: {e}")
    sys.exit(1)

# 测试机器人实例化
try:
    robot = MarvainM6HttpRobot(config)
    print(f"✓ 机器人实例化成功: {robot}")
    print(f"  - 机器人类型: {robot.name}")
    print(f"  - 观测特征数: {len(robot.observation_features)}")
    print(f"  - 动作特征数: {len(robot.action_features)}")
except Exception as e:
    print(f"❌ 机器人实例化失败: {e}")
    sys.exit(1)

# 测试连接（需要 HTTP 服务器运行）
print("\n尝试连接到 HTTP 服务器...")
try:
    robot.connect()
    print(f"✓ 连接成功！")
    print(f"  - 连接状态: {robot.is_connected}")
    print(f"  - 校准状态: {robot.is_calibrated}")
    print(f"  - 发现的相机: {robot._camera_names}")

    # 测试获取观测
    print("\n获取一次观测...")
    obs = robot.get_observation()
    print(f"✓ 获取观测成功，包含 {len(obs)} 个键:")
    for key in sorted(obs.keys()):
        if key.endswith(".pos"):
            print(f"  - {key}: {obs[key]:.2f}°")
        else:
            print(f"  - {key}: shape={obs[key].shape}, dtype={obs[key].dtype}")

    # 测试发送动作
    print("\n发送一个测试动作（全零位置）...")
    action = {f"joint_{i}.pos": 0.0 for i in range(16)}
    actual_action = robot.send_action(action)
    print(f"✓ 动作发送成功")

    # 断开连接
    print("\n断开连接...")
    robot.disconnect()
    print(f"✓ 断开成功，连接状态: {robot.is_connected}")

except Exception as e:
    import traceback
    print(f"❌ 连接或通信失败: {e}")
    print(f"   错误类型: {type(e).__name__}")
    print("\n详细错误信息:")
    traceback.print_exc()
    print("\n请确保 HTTP 服务器正在运行并可访问")
    if robot.is_connected:
        robot.disconnect()
    sys.exit(1)

print("\n" + "="*60)
print("✅ 所有测试通过！HTTP 机器人接口工作正常。")
print("="*60)
