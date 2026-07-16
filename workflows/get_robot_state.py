#!/usr/bin/env python3
"""
快速截取 HTTP 机器人当前状态

用法:
    python workflows/get_robot_state.py
    python workflows/get_robot_state.py --save snapshot.json
    python workflows/get_robot_state.py --url http://192.168.10.100:8010
"""

import argparse
import json
import sys
from datetime import datetime

import requests
import numpy as np


def get_robot_state(http_url='http://192.168.10.123:8010', timeout=5.0):
    """从 HTTP 获取机器人状态"""
    try:
        response = requests.get(f"{http_url}/state", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"❌ 获取失败: {e}")
        sys.exit(1)


def display_state(data):
    """显示状态摘要"""
    print("\n" + "="*60)
    print("HTTP 机器人状态")
    print("="*60)
    print(f"获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"时间戳: {data.get('stamp', 'N/A')}")

    # 关节位置
    if 'joint_states' in data and 'positions' in data['joint_states']:
        joints_rad = data['joint_states']['positions']
        joints_deg = np.degrees(joints_rad).tolist()

        print(f"\n{'─'*60}")
        print("关节位置（14个臂关节）")
        print(f"{'─'*60}")

        print("\n  A臂（左臂）:")
        for i in range(7):
            print(f"    关节 {i}: {joints_rad[i]:8.4f} rad = {joints_deg[i]:7.2f}°")

        print("\n  B臂（右臂）:")
        for i in range(7, 14):
            print(f"    关节 {i}: {joints_rad[i]:8.4f} rad = {joints_deg[i]:7.2f}°")

        # 速度
        if 'velocities' in data['joint_states']:
            velocities = data['joint_states']['velocities']
            max_vel = max(abs(v) for v in velocities)
            print(f"\n  最大速度: {max_vel:.6f} rad/s")

        # 力矩
        if 'efforts' in data['joint_states']:
            efforts = data['joint_states']['efforts']
            print(f"\n  关节力矩:")
            print(f"    A臂: {[round(e, 2) for e in efforts[:7]]}")
            print(f"    B臂: {[round(e, 2) for e in efforts[7:14]]}")

    # 夹爪
    if 'gripper_left' in data and 'gripper_right' in data:
        print(f"\n{'─'*60}")
        print("夹爪位置")
        print(f"{'─'*60}")

        gripper_left = data['gripper_left']
        gripper_right = data['gripper_right']

        if isinstance(gripper_left, list) and len(gripper_left) > 0:
            left_rad = gripper_left[0]
            left_deg = np.degrees(left_rad)
            print(f"  左夹爪: {left_rad:8.4f} rad = {left_deg:7.2f}°")
        else:
            print(f"  左夹爪: {gripper_left}")

        if isinstance(gripper_right, list) and len(gripper_right) > 0:
            right_rad = gripper_right[0]
            right_deg = np.degrees(right_rad)
            print(f"  右夹爪: {right_rad:8.4f} rad = {right_deg:7.2f}°")
        else:
            print(f"  右夹爪: {gripper_right}")

    # 末端执行器
    if 'eef_left' in data or 'eef_right' in data:
        print(f"\n{'─'*60}")
        print("末端执行器")
        print(f"{'─'*60}")

        eef_left = data.get('eef_left')
        eef_right = data.get('eef_right')

        if eef_left and isinstance(eef_left, dict) and 'position' in eef_left:
            pos = eef_left['position']
            print(f"  左臂末端: X={pos[0]:.4f}, Y={pos[1]:.4f}, Z={pos[2]:.4f}")
        else:
            print(f"  左臂末端: {eef_left}")

        if eef_right and isinstance(eef_right, dict) and 'position' in eef_right:
            pos = eef_right['position']
            print(f"  右臂末端: X={pos[0]:.4f}, Y={pos[1]:.4f}, Z={pos[2]:.4f}")
        else:
            print(f"  右臂末端: {eef_right}")

    # 相机
    if 'quad_image' in data:
        quad = data['quad_image']
        print(f"\n{'─'*60}")
        print("相机图像")
        print(f"{'─'*60}")
        print(f"  格式: {quad.get('format', 'N/A')}")
        if 'data' in quad:
            data_len = len(quad['data'])
            size_kb = data_len * 3 // 4 / 1024
            print(f"  Base64 长度: {data_len:,} 字符")
            print(f"  预估大小: {size_kb:.1f} KB")

    print("\n" + "="*60)


def save_state(data, filename):
    """保存状态到文件"""
    # 创建摘要版本（不包含大图像数据）
    summary = {
        'timestamp': datetime.now().isoformat(),
        'stamp': data.get('stamp'),
        'joint_states': data.get('joint_states'),
        'gripper_left': data.get('gripper_left'),
        'gripper_right': data.get('gripper_right'),
        'eef_left': data.get('eef_left'),
        'eef_right': data.get('eef_right'),
    }

    # 添加图像元数据（不保存实际数据）
    if 'quad_image' in data:
        quad = data['quad_image']
        summary['quad_image'] = {
            'format': quad.get('format'),
            'data_size': len(quad.get('data', ''))
        }

    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ 状态已保存到: {filename}")


def main():
    parser = argparse.ArgumentParser(description='获取 HTTP 机器人当前状态')
    parser.add_argument(
        '--url',
        default='http://192.168.10.123:8010',
        help='HTTP 服务器地址'
    )
    parser.add_argument(
        '--save',
        metavar='FILE',
        help='保存到文件（不包含图像数据）'
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=5.0,
        help='请求超时（秒）'
    )

    args = parser.parse_args()

    # 获取状态
    data = get_robot_state(args.url, args.timeout)

    # 显示摘要
    display_state(data)

    # 保存文件
    if args.save:
        save_state(data, args.save)


if __name__ == '__main__':
    main()
