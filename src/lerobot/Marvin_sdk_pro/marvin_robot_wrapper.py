#!/usr/bin/env python3
"""
Marvin Robot Wrapper for LeRobot Integration
包装 Marvin SDK 以适配 LeRobot 的 Robot 接口
"""
import sys
import os
import time
import logging
import math

logging.basicConfig(format='%(message)s')
logger = logging.getLogger('debug_printer')
logger.setLevel(logging.INFO)# 一键关闭所有调试打印
logger.setLevel(logging.DEBUG)  # 默认开启DEBUG级

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from fx_robot import Marvin_Robot, DCSS
from KM_CAN import KMGripperControl, Motor, KM_Motor_Type, Control_Type

logger = logging.getLogger(__name__)


class MarvinRobotWrapper:
    """
    Marvin双臂机器人封装类

    机器人配置:
    - A臂(左臂): 7个关节
    - B臂(右臂): 7个关节
    - 左夹爪: 1个关节 (对应A臂)
    - 右夹爪: 1个关节 (对应B臂)
    - 总共: 16个关节 (7+7+1+1)

    关节顺序: [A臂7关节, B臂7关节, 左夹爪, 右夹爪]
    """

    def __init__(self, robot_ip='192.168.10.190', control_mode='position'):
        """初始化

        Args:
            robot_ip: 机器人IP地址
            control_mode: 控制模式，'position' 或 'impedance'
        """
        self.robot_ip = robot_ip
        self.control_mode = control_mode  # 'position' 或 'impedance'
        self.robot = Marvin_Robot()
        self.dcss = DCSS()
        self._connected = False

        # 夹爪相关
        self._gripper = None
        self._motor_left = None
        self._motor_right = None
        self._gripper_connected = False

    def connect(self) -> bool:
        """连接机器人和夹爪

        Returns:
            bool: 连接是否成功
        """
        logger.info(f"正在连接机器人 {self.robot_ip}...")

        # 1. 连接机器人主体
        init = self.robot.connect(self.robot_ip)
        if init == 0:
            logger.error('机器人连接失败! 端口可能被占用')
            return False

        # 2. 验证连接（检查数据流）
        time.sleep(2.0)

        # ✨ 关键：发送命令激活数据流。
        # 否则 subscribe() 返回的 frame_serial 会一直为 0，
        # 详见 debug_report_frame_serial.md。
        self.robot.clear_set()
        self.robot.log_switch('1')
        self.robot.send_cmd()
        time.sleep(0.5)

        motion_tag = 0
        frame_update = None

        for i in range(5):
            sub_data = self.robot.subscribe(self.dcss)
            frame_serial = sub_data['outputs'][0]['frame_serial']

            if frame_serial != 0 and frame_update != frame_serial:
                motion_tag += 1
                frame_update = frame_serial
            time.sleep(0.2)

        if motion_tag == 0:
            logger.error('机器人数据流未激活（frame_serial 持续为 0）')
            # 主动释放以避免半连接状态影响下次 connect
            try:
                self.robot.release_robot()
            except Exception:
                pass
            return False

        logger.info('✓ 机器人连接成功')
        self._connected = True

        # 开启日志
        self.robot.log_switch('0')
        self.robot.local_log_switch('0')

        # 3. 初始化夹爪
        if not self._init_grippers():
            logger.warning('⚠ 夹爪初始化失败，但机器人主体已连接')
            # 不返回 False，允许机器人在没有夹爪的情况下工作

        # 4. 切换到对应控制模式
        if self.control_mode == 'position':
            self._enable_position_mode()
        elif self.control_mode == 'impedance':
            self._enable_impedance_mode()
        else:
            raise ValueError(f"不支持的控制模式: {self.control_mode}，请使用 'position' 或 'impedance'")

        return True

    def _init_grippers(self) -> bool:
        """初始化双夹爪（内部方法）"""
        try:
            logger.info("初始化双夹爪...")

            # 夹爪配置
            left_gripper_id = 1
            left_gripper_master_id = 0
            right_gripper_id = 2
            right_gripper_master_id = 0

            self._gripper = KMGripperControl(robot=self.robot)
            self._motor_left = Motor(KM_Motor_Type.DM4310, left_gripper_id, left_gripper_master_id)
            self._motor_right = Motor(KM_Motor_Type.DM4310, right_gripper_id, right_gripper_master_id)

            self._gripper.addMotor(self._motor_left)
            self._gripper.add_to_ch(self._motor_left, "left")
            self._gripper.addMotor(self._motor_right)
            self._gripper.add_to_ch(self._motor_right, "right")

            # 禁用再重新使能
            for motor in (self._motor_left, self._motor_right):
                self._gripper.disable(motor)
            time.sleep(0.5)

            # 切换到MIT控制模式并使能
            for motor in (self._motor_left, self._motor_right):
                self._gripper.switchControlMode(motor, Control_Type.MIT)
                self._gripper.enable(motor)
            time.sleep(0.5)

            self._gripper_connected = True
            logger.info("✓ 双夹爪初始化成功")
            return True

        except Exception as e:
            logger.error(f"✗ 夹爪初始化失败: {e}")
            self._gripper = None
            self._gripper_connected = False
            return False

    def _enable_position_mode(self):
        """使能位置控制模式"""
        logger.info("切换到位置控制模式...")

        # 检查当前状态
        sub_data = self.robot.subscribe(self.dcss)
        a_state = sub_data['states'][0]['cur_state']
        b_state = sub_data['states'][1]['cur_state']

        # 如果不是位置模式(state=1)，则切换
        if a_state != 1 or b_state != 1:
            self.robot.clear_set()
            self.robot.set_state(arm='A', state=1)
            self.robot.set_state(arm='B', state=1)
            self.robot.send_cmd()
            time.sleep(1.0)

            # 验证
            sub_data = self.robot.subscribe(self.dcss)
            a_state = sub_data['states'][0]['cur_state']
            b_state = sub_data['states'][1]['cur_state']

            if a_state == 1 and b_state == 1:
                logger.info("✓ 已切换到位置控制模式")
            else:
                logger.warning(f"⚠ 模式切换可能未成功: A={a_state}, B={b_state}")
        else:
            logger.info("✓ 已处于位置控制模式")

    def _enable_impedance_mode(self, K=None, D=None):
        """使能关节阻抗模式

        Args:
            K: 刚度参数 [7个值]，默认 [2,2,2,1,1,1,1]
            D: 阻尼参数 [7个值]，默认 [0.5,0.5,0.5,0.3,0.3,0.3,0.3]
        """
        logger.info("切换到关节阻抗模式...")

        # 默认阻抗参数（柔顺）
        if K is None:
            K = [5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0]
        if D is None:
            D = [0.6, 0.6, 0.6, 0.4, 0.2, 0.2, 0.2]

        # 重力补偿参数
        # kineParams: [x, y, z, rx, ry, rz] 工具相对末端法兰的偏移（mm）和姿态（度）
        # dynamicParams: [m, mx, my, mz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
        kine_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 无偏移
        left_dyn_params = [0.6, 0.0, 0.0, 50.0, 0.004, 0.0, 0.0, 0.004, 0.0, 0.03]
        right_dyn_params = [0.6, 0.0, 0.0, 50.0, 0.004, 0.0, 0.0, 0.004, 0.0, 0.03]

        # 对A臂和B臂都设置阻抗模式
        for arm in ['A', 'B']:
            arm_idx = 0 if arm == 'A' else 1
            dyn_params = left_dyn_params if arm == 'A' else right_dyn_params

            # 检查当前状态
            sub_data = self.robot.subscribe(self.dcss)
            initial_state = sub_data['states'][arm_idx]['cur_state']

            # 如果不在下伺服状态，先下使能
            if initial_state != 0:
                self.robot.clear_set()
                self.robot.set_state(arm=arm, state=0)
                self.robot.send_cmd()
                time.sleep(0.5)

            # 1. 设置重力补偿参数（在扭矩模式前）
            logger.info(f"设置{arm}臂重力补偿参数...")
            self.robot.set_tool(arm=arm, kineParams=kine_params, dynamicParams=dyn_params)
            time.sleep(0.3)

            # 2. 设置扭矩模式
            self.robot.clear_set()
            self.robot.set_state(arm=arm, state=3)  # state=3 扭矩模式
            self.robot.send_cmd()
            time.sleep(0.5)

            # 3. 设置关节阻抗类型
            self.robot.clear_set()
            self.robot.set_impedance_type(arm=arm, type=1)  # type=1 关节阻抗
            self.robot.send_cmd()
            time.sleep(0.3)

            # 4. 设置阻抗参数
            self.robot.clear_set()
            self.robot.set_joint_kd_params(arm=arm, K=K, D=D)
            self.robot.send_cmd()
            time.sleep(0.3)

            # 5. 设置速度和加速度
            self.robot.clear_set()
            self.robot.set_vel_acc(arm=arm, velRatio=10, AccRatio=10)
            self.robot.send_cmd()
            time.sleep(0.3)

        logger.info("✓ 双臂已切换到关节阻抗模式（含重力补偿）")

    def is_connected(self) -> bool:
        """检查是否连接

        Returns:
            bool: 机器人主体和夹爪是否都已连接
        """
        return self._connected and self._gripper_connected

    def get_joint_positions(self) -> list[float]:
        """获取所有关节位置

        Returns:
            list[float]: 16个关节的角度 [A臂7关节, B臂7关节, 左夹爪, 右夹爪]
                        单位: 全部为度 (°)
        """
        if not self._connected:
            raise RuntimeError("机器人未连接")

        sub_data = self.robot.subscribe(self.dcss)

        # 获取A臂和B臂关节位置 (单位: 度)
        a_joints = sub_data['outputs'][0]['fb_joint_pos']  # 7个关节
        b_joints = sub_data['outputs'][1]['fb_joint_pos']  # 7个关节

        # 获取夹爪位置 (单位: 弧度，需转换为度)
        if self._gripper_connected:
            self._gripper.recv()
            left_gripper_rad = self._motor_left.getPosition()   # 弧度
            right_gripper_rad = self._motor_right.getPosition() # 弧度

            # 弧度 → 度
            left_gripper_deg = math.degrees(left_gripper_rad)
            right_gripper_deg = math.degrees(right_gripper_rad)
        else:
            left_gripper_deg = 0.0
            right_gripper_deg = 0.0

        # 拼接成16个关节: [A臂7, B臂7, 左夹爪, 右夹爪]（全部为度）
        positions = a_joints + b_joints + [left_gripper_deg, right_gripper_deg]

        return positions

    def set_joint_positions(self, positions: list[float], vel_ratio: int = 20, acc_ratio: int = 20):
        """设置所有关节位置

        Args:
            positions: 16个关节的目标角度 [A臂7, B臂7, 左夹爪, 右夹爪]
                      单位: 全部为度 (°)
            vel_ratio: 速度百分比 (1-100)
            acc_ratio: 加速度百分比 (1-100)
        """
        if not self._connected:
            raise RuntimeError("机器人未连接")

        if len(positions) != 16:
            raise ValueError(f"需要16个关节位置，得到 {len(positions)} 个")

        # 拆分: 前7个A臂，中7个B臂，后2个夹爪
        a_joints = positions[0:7]
        b_joints = positions[7:14]
        left_gripper_deg = positions[14]
        right_gripper_deg = positions[15]

        # 夹爪: 度 → 弧度
        left_gripper_rad = math.radians(left_gripper_deg)
        right_gripper_rad = math.radians(right_gripper_deg)

        # 1. 设置速度和加速度
        self.robot.clear_set()
        self.robot.set_vel_acc(arm='A', velRatio=vel_ratio, AccRatio=acc_ratio)
        self.robot.set_vel_acc(arm='B', velRatio=vel_ratio, AccRatio=acc_ratio)
        self.robot.send_cmd()
        time.sleep(0.01)

        # 2. 发送机械臂目标位置（根据控制模式）
        self.robot.clear_set()
        self.robot.set_joint_cmd_pose(arm='A', joints=a_joints)
        self.robot.set_joint_cmd_pose(arm='B', joints=b_joints)
        self.robot.send_cmd()

        # 3. 发送夹爪目标位置（使用弧度）
        if self._gripper_connected:
            # 夹爪使用高刚度高阻尼进行精确控制
            stiffness = 8.0
            damping = 0.20

            self._gripper.controlMIT(self._motor_left, stiffness, damping, left_gripper_rad, 0.0, 0.0)
            self._gripper.controlMIT(self._motor_right, stiffness, damping, right_gripper_rad, 0.0, 0.0)

    def disconnect(self):
        """断开连接并下使能"""
        if not self._connected:
            return

        logger.info("断开机器人连接...")

        # 1. 如果是阻抗模式，先退出阻抗/拖动模式
        if self.control_mode == 'impedance':
            logger.info("退出阻抗模式...")
            self.robot.clear_set()
            self.robot.set_drag_space(arm='A', dgType=0)  # 退出拖动
            self.robot.set_drag_space(arm='B', dgType=0)
            self.robot.send_cmd()
            time.sleep(0.3)

        # 2. 下使能夹爪
        if self._gripper_connected and self._gripper is not None:
            logger.info("下使能夹爪...")
            for motor in (self._motor_left, self._motor_right):
                try:
                    self._gripper.disable(motor)
                except Exception as e:
                    logger.warning(f"⚠ 禁用夹爪失败: {e}")
            time.sleep(0.5)

        # 3. 下使能机械臂
        logger.info("下使能机械臂...")
        self.robot.clear_set()
        self.robot.set_state(arm='A', state=0)  # state=0 下伺服
        self.robot.set_state(arm='B', state=0)
        self.robot.send_cmd()
        time.sleep(0.5)

        # 4. 释放连接
        self.robot.release_robot()

        self._connected = False
        self._gripper_connected = False

        logger.info("✓ 已断开连接")


def test_marvin_robot():
    """测试函数

    测试流程:
    1. 连接机械臂和夹爪
    2. 检查连接状态
    3. 读取当前关节位置
    4. 移动最后一个关节 +15°,夹爪开到 38°
    5. 返回初始位置 + 读回核对
    6. 断开连接
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    print("\n" + "="*60)
    print("  Marvin Robot Wrapper 测试")
    print("="*60 + "\n")

    # 选择控制模式: 'position' 或 'impedance'
    control_mode = 'position'  # 改为 'impedance' 可测试阻抗模式
    print(f"控制模式: {control_mode}\n")

    robot = MarvinRobotWrapper(robot_ip='192.168.10.190', control_mode=control_mode)

    try:
        # 1. 连接
        print("步骤 1: 连接机器人和夹爪...")
        if not robot.connect():
            print("✗ 连接失败")
            return
        print("✓ 连接成功\n")

        # 2. 检查连接状态
        print("步骤 2: 检查连接状态...")
        if robot.is_connected():
            print("✓ 机器人和夹爪均已连接\n")
        else:
            print("⚠ 部分设备未连接（可能缺少夹爪）\n")

        # 3. 读取当前位置
        print("步骤 3: 读取当前关节位置...")
        initial_pos = robot.get_joint_positions()
        print(f"A臂 (度): {[round(p, 2) for p in initial_pos[0:7]]}")
        print(f"B臂 (度): {[round(p, 2) for p in initial_pos[7:14]]}")
        print(f"左夹爪 (度): {initial_pos[14]:.2f}")
        print(f"右夹爪 (度): {initial_pos[15]:.2f}\n")

        # 4. 移动：两臂最后一个关节 +15°,夹爪开到 38°
        print("步骤 4: 移动关节...")
        target_pos = initial_pos.copy()

        # A臂最后关节 +15度
        target_pos[6] = initial_pos[6] + 15.0
        # B臂最后关节 +15度
        target_pos[13] = initial_pos[13] + 15.0
        # 左夹爪 → 38°
        target_pos[14] = 38.0
        # 右夹爪 → 38°
        target_pos[15] = 38.0

        print(f"目标位置:")
        print(f"  A臂关节7: {initial_pos[6]:.2f}° → {target_pos[6]:.2f}°")
        print(f"  B臂关节7: {initial_pos[13]:.2f}° → {target_pos[13]:.2f}°")
        print(f"  左夹爪: {initial_pos[14]:.2f}° → {target_pos[14]:.2f}°")
        print(f"  右夹爪: {initial_pos[15]:.2f}° → {target_pos[15]:.2f}°")

        robot.set_joint_positions(target_pos, vel_ratio=20, acc_ratio=20)
        print("✓ 已发送目标位置\n")

        # 等待运动完成
        print("等待运动完成...")
        time.sleep(10.0)

        # 读取实际到达位置
        current_pos = robot.get_joint_positions()
        print(f"实际位置:")
        print(f"  A臂关节7: {current_pos[6]:.2f}°")
        print(f"  B臂关节7: {current_pos[13]:.2f}°")
        print(f"  左夹爪: {current_pos[14]:.2f}°")
        print(f"  右夹爪: {current_pos[15]:.2f}°\n")

        # 5. 返回初始位置 + 读回核对
        print("步骤 5: 返回初始位置...")
        robot.set_joint_positions(initial_pos, vel_ratio=20, acc_ratio=20)
        print("✓ 已发送初始位置\n")

        # 等待运动完成(运动量与 step 4 相当,统一给 8s)
        print("等待返回...")
        time.sleep(8.0)

        # 读回核对
        back_pos = robot.get_joint_positions()
        err = [abs(back_pos[i] - initial_pos[i]) for i in range(16)]
        max_err = max(err)
        print(f"回零误差(度):")
        print(f"  A臂: {[round(e, 2) for e in err[0:7]]}")
        print(f"  B臂: {[round(e, 2) for e in err[7:14]]}")
        print(f"  左夹爪: {err[14]:.2f}°")
        print(f"  右夹爪: {err[15]:.2f}°")
        print(f"  最大误差: {max_err:.2f}°")
        if max_err > 2.0:
            print("⚠ 警告: 回零误差过大(>2°),请检查机械臂或夹爪状态\n")
        else:
            print("✓ 回零精度符合预期\n")

        # 6. 断开连接
        print("步骤 6: 断开连接...")
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
    test_marvin_robot()
