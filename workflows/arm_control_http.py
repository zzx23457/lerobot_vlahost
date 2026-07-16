#!/usr/bin/env python3
"""
机械臂 HTTP 接口简化控制脚本

功能:
1. 查看当前关节位置
2. 移动到指定位置
3. 移动到 home 位置
4. 保存当前位置
5. 回放保存的位置

注意: HTTP 接口不支持阻抗模式、拖动模式、状态查询等高级功能
"""

import sys
import os
import json
import time
import logging
from pathlib import Path

import requests
import numpy as np

# 导入中心配置的 home 位置
sys.path.insert(0, str(Path(__file__).parent))
from _robot_home_config import HOME_LEFT_ARM, get_home_position

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class HTTPArmController:
    """基于 HTTP 接口的机械臂控制器（简化版）"""

    # 双臂镜像规则：索引 0/2/4/6 取反，索引 1/3/5 保持一致
    _MIRROR_INDICES = (0, 2, 4, 6)

    # 左臂（A 臂）标准 home 位置 - 从中心配置导入
    # 注意：这里保留为类属性是为了向后兼容，但实际值来自 _robot_home_config.py
    DEFAULT_HOME_LEFT = HOME_LEFT_ARM

    @classmethod
    def _mirror_joints(cls, joints):
        """按镜像规则生成对侧臂的关节角度"""
        mirrored = list(joints)
        for i in cls._MIRROR_INDICES:
            mirrored[i] = -mirrored[i]
        return mirrored

    @classmethod
    def get_default_home(cls, arm):
        """获取指定臂的标准 home 位置

        Args:
            arm: 'A'（左臂）或 'B'（右臂，由左臂镜像生成）
        """
        if arm == 'A':
            return list(cls.DEFAULT_HOME_LEFT)
        if arm == 'B':
            return cls._mirror_joints(cls.DEFAULT_HOME_LEFT)
        raise ValueError(f"arm 必须是 'A' 或 'B'，得到: {arm}")

    def __init__(self, http_url='http://192.168.10.123:8010'):
        self.http_url = http_url
        self.session = requests.Session()
        self.connected = False
        self.saved_positions = {}  # 保存的位置

    def connect(self):
        """测试连接"""
        logger.info(f"正在连接 HTTP 服务器 {self.http_url}...")
        try:
            response = self.session.get(f"{self.http_url}/state", timeout=5.0)
            response.raise_for_status()
            data = response.json()

            if "joint_states" in data:
                logger.info("✓ 连接成功!")
                self.connected = True
                return True
            else:
                logger.error("✗ 响应格式不正确")
                return False

        except requests.RequestException as e:
            logger.error(f"✗ 连接失败: {e}")
            return False

    def get_current_state(self):
        """获取当前状态（14个臂关节 + 2个夹爪）"""
        if not self.connected:
            logger.error("未连接到服务器")
            return None

        try:
            response = self.session.get(f"{self.http_url}/state", timeout=5.0)
            response.raise_for_status()
            data = response.json()

            # 臂关节（弧度）。服务端 vlahost 当前不返回 joint_states，
            # 缺失时 arm_joints=None（不引入假值），由调用方决定如何降级。
            joint_states = data.get("joint_states") if isinstance(data, dict) else None
            if isinstance(joint_states, dict) and "positions" in joint_states:
                arm_joints_rad = joint_states["positions"]
                if len(arm_joints_rad) == 14:
                    arm_joints_deg = np.degrees(arm_joints_rad).tolist()
                else:
                    arm_joints_deg = None
            else:
                arm_joints_deg = None

            # 夹爪（弧度）。服务端 gripper 字段历史格式：list[number] 或
            # dict {"position": number, "velocity": ..., ...}。两种都兼容，
            # 格式不识别时返回 None（调用方决定如何降级）。
            def _gripper_pos(v):
                if isinstance(v, list):
                    return float(v[0]) if v else None
                if isinstance(v, dict):
                    pos = v.get("position")
                    return float(pos) if pos is not None else None
                if isinstance(v, (int, float)):
                    return float(v)
                return None

            gripper_left_rad = _gripper_pos(data.get("gripper_left")) if isinstance(data, dict) else None
            gripper_right_rad = _gripper_pos(data.get("gripper_right")) if isinstance(data, dict) else None
            gripper_left_deg = float(np.degrees(gripper_left_rad)) if gripper_left_rad is not None else None
            gripper_right_deg = float(np.degrees(gripper_right_rad)) if gripper_right_rad is not None else None

            return {
                'arm_joints': arm_joints_deg,    # 14个 or None
                'gripper_left': gripper_left_deg,
                'gripper_right': gripper_right_deg,
                'all_joints': (arm_joints_deg + [gripper_left_deg, gripper_right_deg]) if arm_joints_deg else None,
            }

        except requests.RequestException as e:
            logger.error(f"获取状态失败: {e}")
            return None

    def send_action(self, arm_joints_deg, gripper_left_deg=0.0, gripper_right_deg=0.0):
        """发送动作指令

        Args:
            arm_joints_deg: 14个臂关节位置（度数）
            gripper_left_deg: 左夹爪位置（度数）
            gripper_right_deg: 右夹爪位置（度数）
        """
        if not self.connected:
            logger.error("未连接到服务器")
            return False

        if len(arm_joints_deg) != 14:
            logger.error(f"需要14个关节，得到{len(arm_joints_deg)}个")
            return False

        try:
            # 转换为弧度并分离左右臂
            arm_joints_rad = np.radians(arm_joints_deg)
            joint_left_rad = arm_joints_rad[:7].tolist()   # 前7个是左臂
            joint_right_rad = arm_joints_rad[7:14].tolist()  # 后7个是右臂

            gripper_left_rad = np.radians(gripper_left_deg)
            gripper_right_rad = np.radians(gripper_right_deg)

            # 服务端 /action 合法字段是 jointcmd_left/right（参见调试面板 JS 和
            # OpenAPI schema）。用错字段名（joint_left/right）会被服务端静默丢弃，
            # 机器人不动——这是之前 home 指令无效的原因。
            payload = {
                "jointcmd_left": joint_left_rad,    # 左臂7个关节
                "jointcmd_right": joint_right_rad,  # 右臂7个关节
                "gripper_left": float(gripper_left_rad),
                "gripper_right": float(gripper_right_rad),
            }

            response = self.session.post(
                f"{self.http_url}/action",
                json=payload,
                timeout=5.0
            )
            response.raise_for_status()
            return True

        except requests.RequestException as e:
            logger.error(f"发送动作失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    logger.error(f"服务器响应: {e.response.json()}")
                except:
                    logger.error(f"服务器响应: {e.response.text[:200]}")
            return False

    def move_to_home(self, arm='B'):
        """移动指定臂到 home 位置（保持另一臂不动）

        Args:
            arm: 'A'（左臂）或 'B'（右臂）
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"移动 {arm} 臂到 home 位置（另一臂保持不动）...")
        logger.info(f"{'='*50}")

        # 获取当前位置
        state = self.get_current_state()
        if state is None:
            return False

        current_joints = state['arm_joints']
        logger.info(f"当前位置:")
        logger.info(f"  A臂: {[round(j, 2) for j in current_joints[:7]]}")
        logger.info(f"  B臂: {[round(j, 2) for j in current_joints[7:14]]}")

        # 准备目标位置：只移动指定臂，另一臂保持当前位置
        if arm == 'A':
            # 移动左臂到home，右臂保持不动
            home_left = self.get_default_home('A')
            target_joints = home_left + current_joints[7:14]
            logger.info(f"目标: A臂→home, B臂→保持")
        elif arm == 'B':
            # 移动右臂到home，左臂保持不动
            home_right = self.get_default_home('B')
            target_joints = current_joints[:7] + home_right
            logger.info(f"目标: A臂→保持, B臂→home")
        else:
            logger.error(f"无效的臂: {arm}，必须是 'A' 或 'B'")
            return False

        logger.info(f"目标位置:")
        logger.info(f"  A臂: {[round(j, 2) for j in target_joints[:7]]}")
        logger.info(f"  B臂: {[round(j, 2) for j in target_joints[7:14]]}")

        # 发送指令
        if self.send_action(target_joints, state['gripper_left'], state['gripper_right']):
            logger.info("✓ 指令已发送")
            logger.info("⚠ 注意: HTTP接口会同时发送两个臂，但只有指定臂会移动")
            return True
        else:
            logger.error("✗ 发送失败")
            return False

    def move_both_arms_to_home(self):
        """同时移动两个臂到 home 位置"""
        logger.info(f"\n{'='*50}")
        logger.info("同时移动两个臂到 home 位置...")
        logger.info(f"{'='*50}")

        # 获取当前位置
        state = self.get_current_state()
        if state is None:
            return False

        current_joints = state['arm_joints']
        logger.info(f"当前位置:")
        logger.info(f"  A臂: {[round(j, 2) for j in current_joints[:7]]}")
        logger.info(f"  B臂: {[round(j, 2) for j in current_joints[7:14]]}")

        # 准备目标位置：两个臂都移动到home
        home_left = self.get_default_home('A')
        home_right = self.get_default_home('B')
        target_joints = home_left + home_right

        logger.info(f"目标位置:")
        logger.info(f"  A臂: {[round(j, 2) for j in home_left]}")
        logger.info(f"  B臂: {[round(j, 2) for j in home_right]}")

        # 发送指令
        if self.send_action(target_joints, state['gripper_left'], state['gripper_right']):
            logger.info("✓ 指令已发送（两个臂同时移动）")
            return True
        else:
            logger.error("✗ 发送失败")
            return False

    def move_to_joints(self, target_joints_14, gripper_left=None, gripper_right=None):
        """移动到指定关节位置

        Args:
            target_joints_14: 14个关节目标位置（度数）
            gripper_left: 左夹爪位置（度数），None保持当前
            gripper_right: 右夹爪位置（度数），None保持当前
        """
        logger.info(f"\n{'='*50}")
        logger.info("移动到指定关节位置...")
        logger.info(f"{'='*50}")

        # 获取当前夹爪位置
        state = self.get_current_state()
        if state is None:
            return False

        if gripper_left is None:
            gripper_left = state['gripper_left']
        if gripper_right is None:
            gripper_right = state['gripper_right']

        logger.info(f"目标臂关节: {[round(j, 2) for j in target_joints_14]}")
        logger.info(f"夹爪: 左={gripper_left:.2f}°, 右={gripper_right:.2f}°")

        if self.send_action(target_joints_14, gripper_left, gripper_right):
            logger.info("✓ 指令已发送")
            return True
        else:
            logger.error("✗ 发送失败")
            return False

    def set_gripper(self, left_deg=None, right_deg=None, iterations=1, hz=30.0):
        """设置夹爪位置（保持臂关节不动），便于隔离测试夹爪控制。

        复用 send_action：把当前臂关节读出来原样回传，只替换目标夹爪值。
        任何 /state 里的变化都可归因于 gripper 命令本身（臂没动）。

        Args:
            left_deg: 左夹爪目标角度（度数），None 保持当前。
            right_deg: 右夹爪目标角度（度数），None 保持当前。
            iterations: 连续发送次数（默认 1；>1 时按 hz 循环）。
            hz: iterations>1 时的发送频率。
        """
        logger.info(f"\n{'='*50}")
        logger.info("设置夹爪位置（保持臂关节不动）...")
        logger.info(f"{'='*50}")

        state = self.get_current_state()
        if state is None:
            return False

        cur_l = state['gripper_left']
        cur_r = state['gripper_right']
        target_l = left_deg  if left_deg  is not None else cur_l
        target_r = right_deg if right_deg is not None else cur_r

        logger.info(f"当前夹爪: 左={cur_l:.2f}°, 右={cur_r:.2f}°")
        logger.info(f"目标夹爪: 左={target_l:.2f}°, 右={target_r:.2f}°")
        logger.info(f"臂关节保持当前: A={[round(j, 2) for j in state['arm_joints'][:7]]}")
        logger.info(f"                  B={[round(j, 2) for j in state['arm_joints'][7:14]]}")
        logger.info(f"发送次数: {iterations} @ {hz} Hz")

        arm = state['arm_joints']
        ok = True
        if iterations <= 1:
            ok = self.send_action(arm, target_l, target_r)
        else:
            dt = 1.0 / hz
            for i in range(iterations):
                ok = self.send_action(arm, target_l, target_r)
                if not ok:
                    logger.error(f"  第 {i+1}/{iterations} 次发送失败，停止")
                    break
                time.sleep(dt)
        if ok:
            logger.info("✓ 夹爪指令已发送")
        else:
            logger.error("✗ 发送失败")
        return ok

    def save_current_position(self, name):
        """保存当前位置"""
        state = self.get_current_state()
        if state is None:
            return False

        self.saved_positions[name] = {
            'arm_joints': state['arm_joints'],
            'gripper_left': state['gripper_left'],
            'gripper_right': state['gripper_right'],
            'timestamp': time.time()
        }

        logger.info(f"✓ 已保存位置 '{name}'")
        return True

    def load_position(self, name):
        """加载保存的位置"""
        if name not in self.saved_positions:
            logger.error(f"位置 '{name}' 不存在")
            logger.info(f"已保存的位置: {list(self.saved_positions.keys())}")
            return False

        pos = self.saved_positions[name]
        logger.info(f"加载位置 '{name}':")
        logger.info(f"  臂关节: {[round(j, 2) for j in pos['arm_joints']]}")
        logger.info(f"  夹爪: 左={pos['gripper_left']:.2f}°, 右={pos['gripper_right']:.2f}°")

        return self.move_to_joints(
            pos['arm_joints'],
            pos['gripper_left'],
            pos['gripper_right']
        )

    def save_positions_to_file(self, filename='saved_positions.json'):
        """保存位置到文件"""
        filepath = Path(filename)
        with open(filepath, 'w') as f:
            json.dump(self.saved_positions, f, indent=2)
        logger.info(f"✓ 位置已保存到文件: {filepath}")

    def load_positions_from_file(self, filename='saved_positions.json'):
        """从文件加载位置"""
        filepath = Path(filename)
        if not filepath.exists():
            logger.warning(f"文件不存在: {filepath}")
            return False

        with open(filepath, 'r') as f:
            self.saved_positions = json.load(f)
        logger.info(f"✓ 已从文件加载 {len(self.saved_positions)} 个位置")
        return True

    def disconnect(self):
        """断开连接"""
        self.session.close()
        self.connected = False
        logger.info("✓ 已断开连接")


def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  机械臂 HTTP 接口控制程序（简化版）")
    print("="*60)

    # 创建控制器
    http_url = input("HTTP 服务器地址 (默认 http://192.168.10.123:8010): ").strip()
    if not http_url:
        http_url = "http://192.168.10.123:8010"

    controller = HTTPArmController(http_url=http_url)

    # 连接
    if not controller.connect():
        print("\n连接失败，程序退出")
        return

    # 尝试加载保存的位置
    controller.load_positions_from_file()

    try:
        while True:
            print("\n" + "-"*60)
            print("请选择操作:")
            print("  1. 查看当前位置（16关节）")
            print("  2. A臂回到 home 位置（B臂保持不动）")
            print("  3. B臂回到 home 位置（A臂保持不动）")
            print("  4. 两个臂同时回到 home 位置")
            print("  5. 移动到自定义位置（输入14个臂关节）")
            print("  6. 保存当前位置")
            print("  7. 加载保存的位置")
            print("  8. 列出所有保存的位置")
            print("  9. 删除保存的位置")
            print(" 10. 保存位置到文件")
            print(" 11. 从文件加载位置")
            print(" 12. 设置左夹爪位置（保持臂和右夹爪不动）")
            print(" 13. 设置右夹爪位置（保持臂和左夹爪不动）")
            print("  0. 退出程序")
            print("-"*60)

            choice = input("请输入选项 (0-13): ").strip()

            if choice == '0':
                break

            elif choice == '1':
                state = controller.get_current_state()
                if state:
                    print(f"\n当前位置 (16关节):")
                    print(f"  A臂 (左臂): {[round(j, 2) for j in state['arm_joints'][:7]]}")
                    print(f"  B臂 (右臂): {[round(j, 2) for j in state['arm_joints'][7:14]]}")
                    print(f"  左夹爪: {state['gripper_left']:.2f}°")
                    print(f"  右夹爪: {state['gripper_right']:.2f}°")

            elif choice == '2':
                controller.move_to_home('A')

            elif choice == '3':
                controller.move_to_home('B')

            elif choice == '4':
                controller.move_both_arms_to_home()

            elif choice == '5':
                print("\n输入14个臂关节角度（度数），用逗号或空格分隔:")
                print("  前7个 = A臂（左臂），后7个 = B臂（右臂）")
                joints_str = input("关节角度: ").strip()

                try:
                    joints = [float(x.strip()) for x in joints_str.replace(',', ' ').split()]
                    if len(joints) == 14:
                        print("\n是否同时设置夹爪? (y/n, 默认n保持当前)")
                        set_gripper = input().strip().lower()

                        gripper_left = None
                        gripper_right = None

                        if set_gripper == 'y':
                            gripper_left = float(input("左夹爪角度（度数）: "))
                            gripper_right = float(input("右夹爪角度（度数）: "))

                        controller.move_to_joints(joints, gripper_left, gripper_right)
                    else:
                        print(f"错误: 需要14个关节，你输入了{len(joints)}个")
                except ValueError:
                    print("输入格式错误，请输入数字")

            elif choice == '6':
                name = input("位置名称: ").strip()
                if name:
                    controller.save_current_position(name)

            elif choice == '7':
                if not controller.saved_positions:
                    print("没有保存的位置")
                else:
                    print("\n已保存的位置:")
                    for i, name in enumerate(controller.saved_positions.keys(), 1):
                        print(f"  {i}. {name}")
                    name = input("输入位置名称: ").strip()
                    if name:
                        controller.load_position(name)

            elif choice == '8':
                if not controller.saved_positions:
                    print("没有保存的位置")
                else:
                    print("\n已保存的位置:")
                    for name, pos in controller.saved_positions.items():
                        print(f"  - {name}")
                        print(f"    A臂: {[round(j, 2) for j in pos['arm_joints'][:7]]}")
                        print(f"    B臂: {[round(j, 2) for j in pos['arm_joints'][7:14]]}")

            elif choice == '9':
                if not controller.saved_positions:
                    print("没有保存的位置")
                else:
                    print("\n已保存的位置:")
                    for i, name in enumerate(controller.saved_positions.keys(), 1):
                        print(f"  {i}. {name}")
                    name = input("输入要删除的位置名称: ").strip()
                    if name in controller.saved_positions:
                        del controller.saved_positions[name]
                        logger.info(f"✓ 已删除位置 '{name}'")
                    else:
                        print(f"位置 '{name}' 不存在")

            elif choice == '10':
                filename = input("文件名 (默认 saved_positions.json): ").strip()
                if not filename:
                    filename = 'saved_positions.json'
                controller.save_positions_to_file(filename)

            elif choice == '11':
                filename = input("文件名 (默认 saved_positions.json): ").strip()
                if not filename:
                    filename = 'saved_positions.json'
                controller.load_positions_from_file(filename)

            elif choice == '12':
                raw = input("左夹爪目标角度 (度): ").strip()
                try:
                    left = float(raw) if raw else None
                except ValueError:
                    print("输入格式错误，请输入数字")
                    continue
                cnt_raw = input("发送次数 (默认 1；>1 时按 30 Hz 循环): ").strip()
                try:
                    iterations = int(cnt_raw) if cnt_raw else 1
                except ValueError:
                    print("输入格式错误，使用默认 1 次")
                    iterations = 1
                controller.set_gripper(left_deg=left, iterations=iterations)

            elif choice == '13':
                raw = input("右夹爪目标角度 (度): ").strip()
                try:
                    right = float(raw) if raw else None
                except ValueError:
                    print("输入格式错误，请输入数字")
                    continue
                cnt_raw = input("发送次数 (默认 1；>1 时按 30 Hz 循环): ").strip()
                try:
                    iterations = int(cnt_raw) if cnt_raw else 1
                except ValueError:
                    print("输入格式错误，使用默认 1 次")
                    iterations = 1
                controller.set_gripper(right_deg=right, iterations=iterations)

            else:
                print("无效选项，请重新输入")

    except KeyboardInterrupt:
        print("\n\n收到中断信号，正在退出...")
    except Exception as e:
        logger.error(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        controller.disconnect()
        print("\n程序已结束\n")


if __name__ == "__main__":
    main()
