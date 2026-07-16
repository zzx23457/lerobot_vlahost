#!/usr/bin/env python3
"""
测试不同控制模式
"""
import sys
import os
import time
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from marvin_robot_wrapper import MarvinRobotWrapper

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def test_mode(mode):
    """测试指定控制模式

    Args:
        mode: 'position' 或 'impedance'
    """
    print("\n" + "="*60)
    print(f"  测试 {mode.upper()} 模式")
    print("="*60 + "\n")

    robot = MarvinRobotWrapper(robot_ip='192.168.10.190', control_mode=mode)

    try:
        # 1. 连接
        print("步骤 1: 连接机器人...")
        if not robot.connect():
            print("✗ 连接失败")
            return
        print("✓ 连接成功\n")

        # 2. 读取当前位置
        print("步骤 2: 读取当前关节位置...")
        initial_pos = robot.get_joint_positions()
        print(f"A臂关节7: {initial_pos[6]:.2f}°")
        print(f"B臂关节7: {initial_pos[13]:.2f}°\n")

        # 3. 移动关节
        print("步骤 3: 移动关节 (各臂最后关节 +15°)...")
        target_pos = initial_pos.copy()
        target_pos[6] = initial_pos[6] + 15.0   # A臂关节7
        target_pos[13] = initial_pos[13] + 15.0 # B臂关节7
        target_pos[14] = 0.2  # 左夹爪
        target_pos[15] = 0.2  # 右夹爪

        robot.set_joint_positions(target_pos, vel_ratio=20, acc_ratio=20)
        print(f"✓ 已发送目标位置")
        print(f"  模式: {mode}")
        print(f"  A臂关节7: {initial_pos[6]:.2f}° → {target_pos[6]:.2f}°")
        print(f"  B臂关节7: {initial_pos[13]:.2f}° → {target_pos[13]:.2f}°\n")

        # 等待3秒
        time.sleep(3.0)

        # 4. 返回初始位置
        print("步骤 4: 返回初始位置...")
        robot.set_joint_positions(initial_pos, vel_ratio=20, acc_ratio=20)
        print("✓ 已发送初始位置\n")

        time.sleep(3.0)

        # 5. 断开连接
        print("步骤 5: 断开连接...")
        robot.disconnect()
        print("✓ 测试完成\n")

    except KeyboardInterrupt:
        print("\n\n收到中断信号...")
        robot.disconnect()
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        robot.disconnect()


if __name__ == "__main__":
    # 从命令行参数选择模式，默认为 position
    mode = sys.argv[1] if len(sys.argv) > 1 else 'position'

    if mode not in ['position', 'impedance']:
        print(f"错误: 不支持的模式 '{mode}'")
        print("用法: python test_control_modes.py [position|impedance]")
        sys.exit(1)

    test_mode(mode)
