#!/usr/bin/env python3
"""从 HTTP 服务器截取当前时刻的机器人状态

实际 API 结构:
- 端点: GET /state (不是 /observation)
- 关节数量: 14 个 (不是 16 个)
- 关节位置: data['joint_states']['positions'] (弧度)
- 图像数据: data['quad_image']['data'] (base64 JPEG)

用法:
    python workflows/robot_interaction/capture_snapshot.py
    python workflows/robot_interaction/capture_snapshot.py --save-images
"""
import argparse
import base64
import json
import sys
from datetime import datetime
from pathlib import Path

import requests
import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("警告: opencv-python 未安装，无法保存图像")


def capture_state(base_url: str = "http://192.168.10.123:8010", timeout: float = 5.0):
    """从 HTTP 服务器获取当前状态"""
    url = f"{base_url}/state"

    print(f"正在从 {url} 获取状态...")
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        print("✓ 成功获取状态数据")
        return data
    except requests.RequestException as e:
        print(f"❌ 连接失败: {e}")
        sys.exit(1)


def rad_to_deg(radians):
    """弧度转角度"""
    return np.degrees(radians)


def save_snapshot(data, save_images=False):
    """保存快照数据"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 提取关节位置
    joints_rad = data['joint_states']['positions']
    joints_deg = rad_to_deg(joints_rad).tolist()

    # 构建保存的数据
    snapshot = {
        'timestamp': timestamp,
        'stamp': data.get('stamp'),
        'joint_count': len(joints_rad),
        'joints': {
            'positions_rad': joints_rad,
            'positions_deg': joints_deg,
            'velocities': data['joint_states'].get('velocities', []),
            'efforts': data['joint_states'].get('efforts', []),
        },
        'end_effector': {
            'left': data.get('eef_left'),
            'right': data.get('eef_right'),
        }
    }

    # 保存 JSON 数据
    json_file = f"snapshot_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(snapshot, f, indent=2)
    print(f"✓ 数据已保存: {json_file}")

    # 显示关节位置
    print(f"\n关节位置 (共 {len(joints_rad)} 个):")
    for i, (rad, deg) in enumerate(zip(joints_rad, joints_deg)):
        print(f"  关节 {i:2d}: {rad:8.4f} rad = {deg:8.2f}°")

    # 保存图像
    if save_images and HAS_OPENCV:
        if 'quad_image' in data and 'data' in data['quad_image']:
            try:
                # 解码 base64 图像
                img_format = data['quad_image'].get('format', 'jpeg')
                img_base64 = data['quad_image']['data']
                img_bytes = base64.b64decode(img_base64)

                # 保存原始图像数据
                img_file = f"snapshot_{timestamp}_quad.{img_format}"
                with open(img_file, 'wb') as f:
                    f.write(img_bytes)
                print(f"✓ 图像已保存: {img_file}")

                # 尝试解码并获取尺寸
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    print(f"  图像尺寸: {img.shape[1]} x {img.shape[0]} ({img.shape[2]} 通道)")

            except Exception as e:
                print(f"⚠️  图像保存失败: {e}")
        else:
            print("⚠️  响应中没有图像数据")

    # 显示末端执行器位置
    if 'eef_left' in data and data['eef_left'] is not None:
        print(f"\n左臂末端位置:")
        if 'position' in data['eef_left']:
            pos = data['eef_left']['position']
            print(f"  X: {pos[0]:.4f}, Y: {pos[1]:.4f}, Z: {pos[2]:.4f}")

    if 'eef_right' in data and data['eef_right'] is not None:
        print(f"\n右臂末端位置:")
        if 'position' in data['eef_right']:
            pos = data['eef_right']['position']
            print(f"  X: {pos[0]:.4f}, Y: {pos[1]:.4f}, Z: {pos[2]:.4f}")

    return json_file


def main():
    parser = argparse.ArgumentParser(
        description="从 HTTP 服务器截取当前机器人状态",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--url",
        default="http://192.168.10.123:8010",
        help="HTTP 服务器地址 (默认: http://192.168.10.123:8010)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="请求超时时间（秒）"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="保存相机图像"
    )

    args = parser.parse_args()

    # 获取状态
    data = capture_state(args.url, args.timeout)

    # 保存快照
    json_file = save_snapshot(data, args.save_images)

    print(f"\n{'='*60}")
    print(f"✅ 快照已保存: {json_file}")
    if args.save_images:
        print(f"   (图像文件已一并保存)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
