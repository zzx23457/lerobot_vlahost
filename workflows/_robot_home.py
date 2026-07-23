#!/usr/bin/env python3
"""_robot_home.py — replay / deploy 共用的"子进程结束后复位 + 下使能"工具

HTTP 接口版本。与 SDK 版本的区别：
1. 通过 HTTP API 发送 home action，而不是直接调用 SDK
2. 单位转换：home 姿态定义为度数，发送时转换为弧度
3. 不支持"下使能"操作（HTTP 接口不暴露电机控制）

任何错误都只打 warning，不抛异常、不影响 caller 的退出码。
"""
import sys
import time
from pathlib import Path

# 30 Hz × 2 秒 = 60 次
_RETURN_HOME_ITERATIONS = 60
_RETURN_HOME_HZ = 30.0


def return_to_home_and_disable(config: dict, returncode: int) -> None:
    """把机械臂送回标准 home 姿态（通过 HTTP 接口）。

    注意：HTTP 接口不支持电机下使能，因此只能发送 home 位置，
    机械臂会保持在该位置（而非断电）。

    Args:
        config: replay / deploy 的配置 dict，需要包含 ``robot.id`` /
            ``robot.http_base_url`` / ``robot.joint_names`` 字段。
        returncode: 子进程退出码；非零时在结尾打一行提示。
    """
    print()
    print("🏠 Return-to-home (HTTP): 把机械臂送回标准 home 姿态...")

    # 1. 从中心配置获取标准 home 姿态（单位：度）-------------------
    sys.path.insert(0, str(Path(__file__).parent))
    from _robot_home_config import get_home_action_16joints

    home_action_deg = get_home_action_16joints()
    home_left = home_action_deg[:7]
    home_right = home_action_deg[7:14]
    home_grippers = home_action_deg[14:16]

    if len(home_action_deg) != 16:
        print(f"   ⚠️  home 姿态长度错误: 期望 16 个关节，实际 {len(home_action_deg)}")
        return

    print(f"   ✓ home 姿态: 左臂={home_left}, 右臂={home_right}, 夹爪={home_grippers}")

    # 2. 重新连机器人（通过 HTTP 接口）-----------------------------
    try:
        # 延迟 import，避免冷启动开销和潜在的循环依赖
        from lerobot.robots import make_robot_from_config

        # 从 config 中获取 joint_names（如果没有则使用默认值）
        joint_names = config.get("robot", {}).get("joint_names")
        if not joint_names:
            # 使用默认 joint_names
            joint_names = [
                "left_arm_joint_1", "left_arm_joint_2", "left_arm_joint_3", "left_arm_joint_4",
                "left_arm_joint_5", "left_arm_joint_6", "left_arm_joint_7",
                "right_arm_joint_1", "right_arm_joint_2", "right_arm_joint_3", "right_arm_joint_4",
                "right_arm_joint_5", "right_arm_joint_6", "right_arm_joint_7",
                "left_gripper", "right_gripper",
            ]

        robot_type = config.get("robot", {}).get("type", "marvain_m6_http")

        if robot_type == "marvain_m6_hybrid":
            # Hybrid: send_action 走 SDK，所以这里也要起一个 hybrid 机器人
            # 让它的 send_action 真正驱动机械臂（HTTP 那条路对 hybrid 是死路）。
            # 临时把 Marvin_sdk_pro 加入 sys.path 让 libMarvinSDK.so 能被 dlopen。
            from lerobot.robots.marvain_m6_hybrid.config_marvain_m6_hybrid import (
                MarvainM6HybridRobotConfig,
            )
            import os
            so_dir = Path(__file__).parent.parent / "src" / "lerobot" / "Marvin_sdk_pro"
            if so_dir.is_dir():
                cur = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = f"{so_dir}:{cur}" if cur else str(so_dir)

            robot_cfg = MarvainM6HybridRobotConfig(
                id=config["robot"]["id"],
                http_base_url=config["robot"].get("http_base_url", "http://192.168.10.123:8010"),
                robot_ip=config["robot"].get("ip", "192.168.10.190"),
                control_mode=config["robot"].get("control_mode", "impedance"),
                timeout=config["robot"].get("timeout", 5.0),
                cameras={},  # 不需要图像
                joint_names=joint_names,
                # 复位完要下使能：让 disconnect() 走完整路径
                # （下使能夹爪 → set_state(0) 下伺服 → release SDK）。
                disable_torque_on_disconnect=True,
            )
            robot = make_robot_from_config(robot_cfg)
            robot.connect()
            print(
                f"   ✓ 已重连机器人 {config['robot']['id']} "
                f"(hybrid: HTTP @ {robot_cfg.http_base_url}, SDK @ {robot_cfg.robot_ip})"
            )
        else:
            # 原 HTTP-only 路径
            from lerobot.robots.marvain_m6_http.config_marvain_m6_http import (
                MarvainM6HttpRobotConfig,
            )
            robot_cfg = MarvainM6HttpRobotConfig(
                id=config["robot"]["id"],
                http_base_url=config["robot"].get("http_base_url", "http://192.168.10.123:8010"),
                timeout=config["robot"].get("timeout", 5.0),
                cameras={},  # 不需要图像
                joint_names=joint_names,
            )
            robot = make_robot_from_config(robot_cfg)
            robot.connect()
            print(f"   ✓ 已重连机器人 {config['robot']['id']} @ {robot_cfg.http_base_url}")
    except Exception as e:
        print(f"   ⚠️  连机器人失败，跳过 return-to-home: {type(e).__name__}: {e}")
        return

    # 3. 尝试使用新 go_home() 方法（如果支持）--------------------
    # 优先使用 go_home() ROS 服务调用（可靠、快速），如果失败
    # 则回退到旧方法（连续发送关节位置动作）。
    try:
        go_home_success = False
        if hasattr(robot, "go_home"):
            try:
                print("   🚀 使用 POST /go_home ROS 服务...")
                go_home_success = robot.go_home()
                if go_home_success:
                    print("   ✓ go_home() 调用成功")
                else:
                    print("   ⚠️  go_home() 返回失败，回退到发送动作模式")
            except Exception as e:
                print(f"   ⚠️  go_home() 调用异常，回退到发送动作模式: {type(e).__name__}: {e}")

        # 4. 回退方案：连续发送 home action 约 2 秒 --------------------
        if not go_home_success:
            # 组装 home action（按 joint_names 顺序）
            home_action: dict[str, float] = {}
            for i, name in enumerate(joint_names):
                home_action[f"{name}.pos"] = float(home_action_deg[i])

            print(f"   📤 回退模式：连续发送 home action ({_RETURN_HOME_ITERATIONS} 次)")
            dt = 1.0 / _RETURN_HOME_HZ
            for _ in range(_RETURN_HOME_ITERATIONS):
                robot.send_action(home_action)
                time.sleep(dt)
            print(f"   ✓ 已发送 {_RETURN_HOME_ITERATIONS} 次 home action")
    except Exception as e:
        print(f"   ⚠️  执行 home 操作失败: {type(e).__name__}: {e}")
    finally:
        # 5. 断连
        try:
            robot.disconnect()
            if robot_type == "marvain_m6_hybrid":
                # Hybrid 走 SDK 完整路径：disable_torque_on_disconnect=True
                # 让 disconnect 真的下使能（夹爪 disable + set_state(0) 下伺服）
                print("   ✓ 机械臂已下使能（home 位置断电）")
            else:
                # HTTP-only: HTTP 接口不支持电机下使能
                print("   ✓ 机械臂已断开连接（保持在 home 位置）")
                print("   ℹ️  注意：HTTP 接口不支持电机下使能，机械臂仍保持通电状态")
        except Exception as e:
            print(f"   ⚠️  断连失败: {type(e).__name__}: {e}")

    # 给用户一个对比：子进程退出码非零时提醒一下
    if returncode != 0:
        print(f"   ℹ️  子进程退出码 {returncode}（非零），home 已送但请检查上面的日志")
