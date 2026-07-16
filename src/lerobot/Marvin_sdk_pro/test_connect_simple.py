#!/usr/bin/env python3
"""
简化版连接测试 - 跳过frame_serial验证
"""
import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from fx_robot import Marvin_Robot, DCSS

print("=" * 60)
print("简化版机器人连接测试")
print("=" * 60)

robot = Marvin_Robot()
dcss = DCSS()

# 连接
print("\n1. 连接机器人 192.168.15.190...")
init = robot.connect('192.168.15.190')
print(f"   connect() 返回值: {init}")

if init == 0:
    print("✗ 连接失败 (端口被占用或网络问题)")
    sys.exit(1)

print("✓ 底层连接成功")

# 等待并读取状态
time.sleep(1)
print("\n2. 读取机器人状态...")

sub_data = robot.subscribe(dcss)

print("\n=== A臂 ===")
print(f"  当前状态: {sub_data['states'][0]['cur_state']}")
print(f"  错误码: {sub_data['states'][0]['err_code']}")
print(f"  frame_serial: {sub_data['outputs'][0]['frame_serial']}")

print("\n=== B臂 ===")
print(f"  当前状态: {sub_data['states'][1]['cur_state']}")
print(f"  错误码: {sub_data['states'][1]['err_code']}")
print(f"  frame_serial: {sub_data['outputs'][1]['frame_serial']}")

# 尝试发送一个命令看是否能激活
print("\n3. 尝试发送命令激活数据流...")
robot.clear_set()
robot.log_switch('1')
robot.send_cmd()
time.sleep(0.5)

# 再次检查
print("\n4. 再次检查 frame_serial...")
for i in range(5):
    sub_data = robot.subscribe(dcss)
    fs_a = sub_data['outputs'][0]['frame_serial']
    fs_b = sub_data['outputs'][1]['frame_serial']
    print(f"   第{i+1}次: A臂={fs_a}, B臂={fs_b}")
    if fs_a != 0 or fs_b != 0:
        print("✓ 数据帧开始更新!")
        break
    time.sleep(0.2)

if fs_a == 0 and fs_b == 0:
    print("\n⚠ frame_serial 仍为0，可能原因:")
    print("  1. 机器人控制器程序未运行")
    print("  2. 机器人处于急停状态")
    print("  3. 机器人需要通过示教器先启动")
    print("\n但底层连接正常，可以尝试继续使用")

robot.release_robot()
print("\n测试完成")
