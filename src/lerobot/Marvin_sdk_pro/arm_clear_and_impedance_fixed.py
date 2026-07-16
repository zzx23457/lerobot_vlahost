#!/usr/bin/env python3
"""
机械臂清错和阻抗/拖动模式设置脚本

功能:
1. 检查并清除机械臂错误
2. 设置阻抗模式（关节阻抗或笛卡尔阻抗）
3. 设置拖动模式
4. 检查限位问题
"""

import sys
import os
import time
import logging

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from fx_robot import Marvin_Robot, DCSS

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class ArmController:
    """机械臂控制器类"""

    # 双臂镜像规则：索引 0/2/4/6 取反，索引 1/3/5 保持一致
    # 镜像源是左臂（A 臂），右臂（B 臂）由左臂镜像生成
    _MIRROR_INDICES = (0, 2, 4, 6)

    # 左臂（A 臂）标准 home 位置 — 取自数据集 frame 0 前 7 维
    # 数据集 observation.state 排列: [A臂(左臂) 7维, B臂(右臂) 7维, 夹爪 2维]
    # DEFAULT_HOME_LEFT = [
    #     66.04866790771484,
    #     -18.997726440429688,
    #     -80.62322998046875,
    #     -84.70333862304688,
    #     -47.016021728515625,
    #     31.47335433959961,
    #     -40.16086959838867,
    # ]
    DEFAULT_HOME_LEFT = [
    97.42, -62.95, -62.8, -114.38, -21.22, 7.35, 31.64
    ]
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
            arm: 'A'（左臂，源数据）或 'B'（右臂，由左臂镜像生成）
        """
        if arm == 'A':
            return list(cls.DEFAULT_HOME_LEFT)
        if arm == 'B':
            return cls._mirror_joints(cls.DEFAULT_HOME_LEFT)
        raise ValueError(f"arm 必须是 'A' 或 'B'，得到: {arm}")

    def __init__(self, robot_ip='192.168.10.190'):
        self.robot = Marvin_Robot()
        self.dcss = DCSS()
        self.robot_ip = robot_ip
        self.connected = False
        self._gripper = None
        self._motor_left = None
        self._motor_right = None

    def connect(self):
        """连接机器人"""
        logger.info(f"正在连接机器人 {self.robot_ip}...")
        init = self.robot.connect(self.robot_ip)

        if init == 0:
            logger.error('连接失败! 端口可能被占用')
            return False

        # 验证连接
        time.sleep(1.0)  # 增加等待时间让共享内存数据同步

        # 发送命令激活数据流（修复：参考test_connect_simple.py）
        logger.info("发送命令激活数据流...")
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
            time.sleep(0.2)  # 增加检查间隔

        if motion_tag > 0:
            logger.info('✓ 机器人连接成功!')
            self.connected = True
            # 开启日志
            self.robot.log_switch('0')
            self.robot.local_log_switch('0')

            return True
        else:
            logger.error('✗ 机器人连接失败!')
            return False

    def check_and_clear_errors(self, arm='B'):
        """检查并清除错误

        Args:
            arm: 'A' 或 'B'
        """
        if not self.connected:
            logger.error("未连接到机器人")
            return False

        logger.info(f"\n{'='*50}")
        logger.info(f"检查 {arm} 臂状态和错误码...")
        logger.info(f"{'='*50}")

        # 获取当前状态
        sub_data = self.robot.subscribe(self.dcss)
        arm_idx = 0 if arm == 'A' else 1

        cur_state = sub_data['states'][arm_idx]['cur_state']
        cmd_state = sub_data['states'][arm_idx]['cmd_state']
        err_code = sub_data['states'][arm_idx]['err_code']

        state_names = {
            0: "下伺服 (IDLE)",
            1: "位置跟随 (POSITION)",
            2: "PVT模式",
            3: "扭矩模式 (TORQUE)",
            4: "协作释放 (RELEASE)"
        }

        logger.info(f"当前状态: {cur_state} - {state_names.get(cur_state, '未知')}")
        logger.info(f"指令状态: {cmd_state}")
        logger.info(f"错误码: {err_code}")

        # 获取伺服错误码
        servo_errors = self.robot.get_servo_error_code(arm)
        logger.info(f"伺服错误码 (7个关节): {servo_errors}")

        # 检查是否有错误
        has_error = False
        if err_code != 0:
            has_error = True
            logger.warning(f"⚠ {arm} 臂存在错误码: {err_code}")

        for i, err in enumerate(servo_errors):
            if err != '0X0':
                has_error = True
                logger.warning(f"⚠ 关节 {i+1} 存在伺服错误: {err}")

        if has_error:
            logger.info(f"正在清除 {arm} 臂错误...")
            self.robot.clear_set()
            self.robot.clear_error(arm)
            self.robot.send_cmd()
            time.sleep(1)

            # 再次检查
            sub_data = self.robot.subscribe(self.dcss)
            new_err_code = sub_data['states'][arm_idx]['err_code']
            new_servo_errors = self.robot.get_servo_error_code(arm)

            logger.info(f"清错后错误码: {new_err_code}")
            logger.info(f"清错后伺服错误: {new_servo_errors}")

            if new_err_code == 0:
                logger.info(f"✓ {arm} 臂错误已清除")
                return True
            else:
                logger.error(f"✗ {arm} 臂仍有错误，可能需要手动检查")
                return False
        else:
            logger.info(f"✓ {arm} 臂无错误")
            return True

    def wait_for_state(self, arm, target_state, timeout=5.0, check_interval=0.1):
        """等待机械臂进入目标状态

        Args:
            arm: 'A' 或 'B'
            target_state: 目标状态码
            timeout: 超时时间（秒）
            check_interval: 检查间隔（秒）

        Returns:
            bool: 是否成功进入目标状态
        """
        arm_idx = 0 if arm == 'A' else 1
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            sub_data = self.robot.subscribe(self.dcss)
            cur_state = sub_data['states'][arm_idx]['cur_state']

            if cur_state == target_state:
                return True

            logger.debug(f"等待状态切换... 当前: {cur_state}, 目标: {target_state}")
            time.sleep(check_interval)

        return False

    def enter_joint_impedance_mode(self, arm='B', K=None, D=None):
        """进入关节阻抗模式

        Args:
            arm: 'A' 或 'B'
            K: 刚度参数 [7个值], 默认 [2,2,2,1,1,1,1]
            D: 阻尼参数 [7个值], 默认 [0.5,0.5,0.5,0.3,0.3,0.3,0.3]
        """
        if not self.connected:
            logger.error("未连接到机器人")
            return False

        # 默认参数（低刚度低阻尼，适合拖动）
        if K is None:
            K = [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]
        if D is None:
            D = [0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3]

        logger.info(f"\n{'='*50}")
        logger.info(f"设置 {arm} 臂进入关节阻抗模式...")
        logger.info(f"{'='*50}")

        arm_idx = 0 if arm == 'A' else 1

        # 检查当前状态
        sub_data = self.robot.subscribe(self.dcss)
        initial_state = sub_data['states'][arm_idx]['cur_state']
        logger.info(f"初始状态: {initial_state}")

        # 如果不在下伺服状态，先下使能
        if initial_state != 0:
            logger.warning(f"当前状态不是下伺服(0)，先下使能...")
            self.robot.clear_set()
            self.robot.set_state(arm=arm, state=0)
            self.robot.send_cmd()
            if not self.wait_for_state(arm, 0, timeout=3.0):
                logger.error(f"无法下使能 {arm} 臂")
                return False
            logger.info("✓ 已下使能")
            time.sleep(0.5)

        # 1. 设置扭矩模式
        logger.info("步骤 1: 切换到扭矩模式...")
        self.robot.clear_set()
        self.robot.set_state(arm=arm, state=3)  # state=3 扭矩模式
        self.robot.send_cmd()

        # 等待状态切换
        if not self.wait_for_state(arm, 3, timeout=5.0):
            sub_data = self.robot.subscribe(self.dcss)
            cur_state = sub_data['states'][arm_idx]['cur_state']
            logger.error(f"✗ 切换到扭矩模式失败，当前状态: {cur_state}")

            # 检查是否有错误
            err_code = sub_data['states'][arm_idx]['err_code']
            if err_code != 0:
                logger.error(f"  错误码: {err_code}")
                servo_errors = self.robot.get_servo_error_code(arm)
                logger.error(f"  伺服错误: {servo_errors}")

            return False

        logger.info("✓ 已进入扭矩模式")
        time.sleep(0.3)

        # 2. 设置关节阻抗类型
        logger.info("步骤 2: 设置关节阻抗类型...")
        self.robot.clear_set()
        self.robot.set_impedance_type(arm=arm, type=1)  # type=1 关节阻抗
        self.robot.send_cmd()
        time.sleep(0.5)

        # 3. 设置阻抗参数
        logger.info(f"步骤 3: 设置阻抗参数...")
        logger.info(f"  刚度 K: {K}")
        logger.info(f"  阻尼 D: {D}")
        self.robot.clear_set()
        self.robot.set_joint_kd_params(arm=arm, K=K, D=D)
        self.robot.send_cmd()
        time.sleep(0.5)

        # 4. 设置速度和加速度
        logger.info("步骤 4: 设置速度和加速度为10%...")
        self.robot.clear_set()
        self.robot.set_vel_acc(arm=arm, velRatio=10, AccRatio=10)
        self.robot.send_cmd()
        time.sleep(0.5)

        # 验证设置
        sub_data = self.robot.subscribe(self.dcss)
        cur_state = sub_data['states'][arm_idx]['cur_state']
        imp_type = sub_data['inputs'][arm_idx]['imp_type']

        if cur_state == 3 and imp_type == 1:
            logger.info(f"✓ {arm} 臂已成功进入关节阻抗模式")
            logger.info(f"  当前状态: {cur_state} (扭矩模式)")
            logger.info(f"  阻抗类型: {imp_type} (关节阻抗)")
            return True
        else:
            logger.error(f"✗ {arm} 臂未能进入关节阻抗模式")
            logger.error(f"  当前状态: {cur_state}, 阻抗类型: {imp_type}")
            return False

    def enter_cart_impedance_mode(self, arm='B', K=None, D=None):
        """进入笛卡尔阻抗模式

        Args:
            arm: 'A' 或 'B'
            K: 刚度参数 [7个值], 默认 [2000,2000,2000,40,40,40,20]
            D: 阻尼参数 [7个值], 默认 [0.1,0.1,0.1,0.3,0.3,0.3,1]
        """
        if not self.connected:
            logger.error("未连接到机器人")
            return False

        # 默认参数
        if K is None:
            K = [2000.0, 2000.0, 2000.0, 40.0, 40.0, 40.0, 20.0]
        if D is None:
            D = [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 1.0]

        logger.info(f"\n{'='*50}")
        logger.info(f"设置 {arm} 臂进入笛卡尔阻抗模式...")
        logger.info(f"{'='*50}")

        arm_idx = 0 if arm == 'A' else 1

        # 检查当前状态
        sub_data = self.robot.subscribe(self.dcss)
        initial_state = sub_data['states'][arm_idx]['cur_state']
        logger.info(f"初始状态: {initial_state}")

        # 如果不在下伺服状态，先下使能
        if initial_state != 0:
            logger.warning(f"当前状态不是下伺服(0)，先下使能...")
            self.robot.clear_set()
            self.robot.set_state(arm=arm, state=0)
            self.robot.send_cmd()
            if not self.wait_for_state(arm, 0, timeout=3.0):
                logger.error(f"无法下使能 {arm} 臂")
                return False
            logger.info("✓ 已下使能")
            time.sleep(0.5)

        # 1. 设置扭矩模式
        logger.info("步骤 1: 切换到扭矩模式...")
        self.robot.clear_set()
        self.robot.set_state(arm=arm, state=3)  # state=3 扭矩模式
        self.robot.send_cmd()

        # 等待状态切换
        if not self.wait_for_state(arm, 3, timeout=5.0):
            sub_data = self.robot.subscribe(self.dcss)
            cur_state = sub_data['states'][arm_idx]['cur_state']
            logger.error(f"✗ 切换到扭矩模式失败，当前状态: {cur_state}")
            return False

        logger.info("✓ 已进入扭矩模式")
        time.sleep(0.3)

        # 2. 设置笛卡尔阻抗类型
        logger.info("步骤 2: 设置笛卡尔阻抗类型...")
        self.robot.clear_set()
        self.robot.set_impedance_type(arm=arm, type=2)  # type=2 坐标阻抗
        self.robot.send_cmd()
        time.sleep(0.5)

        # 3. 设置阻抗参数
        logger.info(f"步骤 3: 设置阻抗参数...")
        logger.info(f"  刚度 K: {K}")
        logger.info(f"  阻尼 D: {D}")
        self.robot.clear_set()
        self.robot.set_cart_kd_params(arm=arm, K=K, D=D, type=2)
        self.robot.send_cmd()
        time.sleep(0.5)

        # 4. 设置速度和加速度
        logger.info("步骤 4: 设置速度和加速度为10%...")
        self.robot.clear_set()
        self.robot.set_vel_acc(arm=arm, velRatio=10, AccRatio=10)
        self.robot.send_cmd()
        time.sleep(0.5)

        # 验证设置
        sub_data = self.robot.subscribe(self.dcss)
        cur_state = sub_data['states'][arm_idx]['cur_state']
        imp_type = sub_data['inputs'][arm_idx]['imp_type']

        if cur_state == 3 and imp_type == 2:
            logger.info(f"✓ {arm} 臂已成功进入笛卡尔阻抗模式")
            logger.info(f"  当前状态: {cur_state} (扭矩模式)")
            logger.info(f"  阻抗类型: {imp_type} (笛卡尔阻抗)")
            return True
        else:
            logger.error(f"✗ {arm} 臂未能进入笛卡尔阻抗模式")
            logger.error(f"  当前状态: {cur_state}, 阻抗类型: {imp_type}")
            return False

    def enter_drag_mode(self, arm='B', drag_type=1):
        """进入拖动模式

        Args:
            arm: 'A' 或 'B'
            drag_type: 拖动类型
                1: 关节空间拖动
                2: 笛卡尔空间x方向拖动
                3: 笛卡尔空间y方向拖动
                4: 笛卡尔空间z方向拖动
                5: 笛卡尔空间旋转方向拖动
        """
        if not self.connected:
            logger.error("未连接到机器人")
            return False

        drag_names = {
            1: "关节空间拖动",
            2: "笛卡尔X方向拖动",
            3: "笛卡尔Y方向拖动",
            4: "笛卡尔Z方向拖动",
            5: "笛卡尔旋转拖动"
        }

        logger.info(f"\n{'='*50}")
        logger.info(f"设置 {arm} 臂进入拖动模式: {drag_names.get(drag_type, '未知')}")
        logger.info(f"{'='*50}")

        # 先进入关节阻抗模式（拖动的前提）
        if not self.enter_joint_impedance_mode(arm=arm):
            logger.error("无法进入关节阻抗模式，拖动设置失败")
            return False

        # 设置拖动空间
        logger.info(f"设置拖动类型为: {drag_names.get(drag_type, '未知')}")
        self.robot.clear_set()
        self.robot.set_drag_space(arm=arm, dgType=drag_type)
        self.robot.send_cmd()
        time.sleep(0.5)

        # 验证
        sub_data = self.robot.subscribe(self.dcss)
        arm_idx = 0 if arm == 'A' else 1

        drag_sp_type = sub_data['inputs'][arm_idx]['drag_sp_type']

        if drag_sp_type == drag_type:
            logger.info(f"✓ {arm} 臂已成功进入拖动模式")
            logger.info(f"  拖动类型: {drag_sp_type} - {drag_names.get(drag_type, '未知')}")
            logger.info("\n⚠ 提示: 可以开始手动拖动机械臂了!")
            return True
        else:
            logger.error(f"✗ {arm} 臂未能进入拖动模式")
            logger.error(f"  期望拖动类型: {drag_type}, 实际: {drag_sp_type}")
            return False

    def exit_drag_mode(self, arm='B'):
        """退出拖动模式"""
        if not self.connected:
            logger.error("未连接到机器人")
            return False

        logger.info(f"退出 {arm} 臂拖动模式...")
        self.robot.clear_set()
        self.robot.set_drag_space(arm=arm, dgType=0)  # 0 = 退出拖动
        self.robot.send_cmd()
        time.sleep(0.5)

        logger.info(f"✓ {arm} 臂已退出拖动模式")
        return True

    def disable_arm(self, arm='B'):
        """下使能机械臂"""
        if not self.connected:
            logger.error("未连接到机器人")
            return False

        logger.info(f"下使能 {arm} 臂...")
        self.robot.clear_set()
        self.robot.set_state(arm=arm, state=0)  # state=0 下伺服
        self.robot.send_cmd()
        time.sleep(0.5)

        logger.info(f"✓ {arm} 臂已下使能")
        return True

    def disable_both_arms(self):
        """同时下使能两个臂（下电前先检查并清除错误）"""
        if not self.connected:
            logger.error("未连接到机器人")
            return False

        logger.info("\n" + "="*50)
        logger.info("同时下使能A臂和B臂...")
        logger.info("="*50)

        # 1. 检查两个臂是否有错误（只检查主错误码）
        sub_data = self.robot.subscribe(self.dcss)
        a_err_code = sub_data['states'][0]['err_code']
        b_err_code = sub_data['states'][1]['err_code']

        has_error = (a_err_code != 0) or (b_err_code != 0)

        if has_error:
            logger.warning(f"⚠ 检测到错误 - A臂: {a_err_code}, B臂: {b_err_code}")
            logger.info("正在清除错误...")

            # 清除A臂错误
            self.robot.clear_set()
            self.robot.clear_error('A')
            self.robot.send_cmd()
            time.sleep(0.3)

            # 清除B臂错误
            self.robot.clear_set()
            self.robot.clear_error('B')
            self.robot.send_cmd()
            time.sleep(0.3)

            # 再次检查
            sub_data = self.robot.subscribe(self.dcss)
            a_err_code = sub_data['states'][0]['err_code']
            b_err_code = sub_data['states'][1]['err_code']

            if a_err_code == 0 and b_err_code == 0:
                logger.info("✓ 错误已清除")
            else:
                logger.warning(f"⚠ 清错后仍有错误 - A臂: {a_err_code}, B臂: {b_err_code}")
        else:
            logger.info("✓ 两个臂均无错误，直接下使能")

        # 2. 下使能两个臂
        logger.info("正在下使能两个臂...")
        self.robot.clear_set()
        self.robot.set_state(arm='A', state=0)  # state=0 下伺服
        self.robot.set_state(arm='B', state=0)
        self.robot.send_cmd()
        time.sleep(0.5)

        # 3. 验证状态
        sub_data = self.robot.subscribe(self.dcss)
        a_state = sub_data['states'][0]['cur_state']
        b_state = sub_data['states'][1]['cur_state']

        if a_state == 0 and b_state == 0:
            logger.info("✓ A臂和B臂已成功下使能")
            logger.info(f"  A臂状态: {a_state} (下伺服)")
            logger.info(f"  B臂状态: {b_state} (下伺服)")
            return True
        else:
            logger.warning("⚠ 下使能可能未完全成功")
            logger.warning(f"  A臂状态: {a_state}")
            logger.warning(f"  B臂状态: {b_state}")
            return False

    def get_current_joint_positions(self, arm='B'):
        """获取当前关节位置"""
        if not self.connected:
            logger.error("未连接到机器人")
            return None

        sub_data = self.robot.subscribe(self.dcss)
        arm_idx = 0 if arm == 'A' else 1

        joint_pos = sub_data['outputs'][arm_idx]['fb_joint_pos']
        return joint_pos

    def init_gripper(self):
        """初始化双夹爪"""
        if not self.connected:
            logger.error("未连接到机器人")
            return False

        try:
            from KM_CAN import KMGripperControl, Motor, KM_Motor_Type, Control_Type

            logger.info(f"\n{'='*50}")
            logger.info("初始化双夹爪...")
            logger.info(f"{'='*50}")

            # 夹爪配置（默认值）
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

            logger.info("步骤 1: 禁用夹爪电机...")
            for motor in (self._motor_left, self._motor_right):
                self._gripper.disable(motor)
            time.sleep(0.5)

            logger.info("步骤 2: 切换到MIT控制模式并使能...")
            for motor in (self._motor_left, self._motor_right):
                self._gripper.switchControlMode(motor, Control_Type.MIT)
                self._gripper.enable(motor)
            time.sleep(0.5)

            logger.info("✓ 双夹爪初始化成功")
            return True
        except Exception as e:
            logger.error(f"✗ 夹爪初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self._gripper = None
            return False

    def disable_gripper(self):
        """下使能双夹爪"""
        if not self.connected:
            logger.error("未连接到机器人")
            return False

        if self._gripper is None:
            logger.error("夹爪未初始化")
            return False

        logger.info(f"\n{'='*50}")
        logger.info("下使能双夹爪...")
        logger.info(f"{'='*50}")

        for motor in (self._motor_left, self._motor_right):
            try:
                self._gripper.disable(motor)
            except Exception as e:
                logger.warning(f"⚠ 禁用电机失败: {e}")

        time.sleep(0.5)
        logger.info("✓ 双夹爪已下使能")
        return True

    def move_gripper_to_position(self, gripper='left', target_angle=0.0, stiffness=8.0, damping=0.20):
        """移动夹爪到指定角度

        Args:
            gripper: 'left' 或 'right'（对应A臂和B臂）
            target_angle: 目标角度（弧度）
            stiffness: 刚度（默认8.0，高刚度精确控制）
            damping: 阻尼（默认0.20）
        """
        if not self.connected:
            logger.error("未连接到机器人")
            return False

        if self._gripper is None:
            logger.error("夹爪未初始化")
            return False

        motor = self._motor_left if gripper == 'left' else self._motor_right
        gripper_name = "左夹爪(A臂)" if gripper == 'left' else "右夹爪(B臂)"

        logger.info(f"\n{'='*50}")
        logger.info(f"移动{gripper_name}到目标角度...")
        logger.info(f"{'='*50}")

        # 读取当前位置
        self._gripper.recv()
        current_pos = motor.getPosition()
        logger.info(f"当前位置: {current_pos:.4f} rad")
        logger.info(f"目标位置: {target_angle:.4f} rad")
        logger.info(f"刚度: {stiffness}, 阻尼: {damping}")

        # 发送控制命令
        self._gripper.controlMIT(
            motor,
            stiffness,
            damping,
            target_angle,
            0.0,
            0.0
        )

        # 等待到达目标位置
        max_wait_time = 3.0
        start_time = time.time()
        reached = False

        while (time.time() - start_time) < max_wait_time:
            self._gripper.recv()
            current_pos = motor.getPosition()
            error = abs(current_pos - target_angle)

            logger.info(f"  当前位置: {current_pos:.4f} rad | 误差: {error:.4f} rad")

            if error < 0.01:  # 误差小于0.01弧度
                reached = True
                break

            time.sleep(0.1)

        if reached:
            logger.info(f"✓ {gripper_name}已到达目标位置")
            return True
        else:
            logger.warning(f"⚠ {gripper_name}未能在{max_wait_time}秒内到达目标位置")
            logger.warning(f"  最终位置: {current_pos:.4f} rad，目标: {target_angle:.4f} rad")
            return False

    def move_to_home_position(self, arm='B', home_joints=None, vel_ratio=10, acc_ratio=10):
        """移动到home位置（关节复位）

        Args:
            arm: 'A'（左臂）或 'B'（右臂）
            home_joints: 目标关节角度 [7个值]，默认使用该臂的标准home姿态
                        （A 来自 DEFAULT_HOME_LEFT，B 由 A 镜像生成）
            vel_ratio: 速度百分比 (1-100)
            acc_ratio: 加速度百分比 (1-100)
        """
        if not self.connected:
            logger.error("未连接到机器人")
            return False

        # 默认 home 位置：A 来自左臂实测值，B 由 A 镜像得出
        if home_joints is None:
            home_joints = self.get_default_home(arm)

        logger.info(f"\n{'='*50}")
        logger.info(f"移动 {arm} 臂到home位置（关节复位）...")
        logger.info(f"{'='*50}")
        logger.info(f"目标关节位置: {home_joints}")

        arm_idx = 0 if arm == 'A' else 1

        # 检查当前状态
        sub_data = self.robot.subscribe(self.dcss)
        initial_state = sub_data['states'][arm_idx]['cur_state']
        current_joints = sub_data['outputs'][arm_idx]['fb_joint_pos']

        logger.info(f"当前状态: {initial_state}")
        logger.info(f"当前关节位置: {[round(j, 2) for j in current_joints]}")

        # 如果不在下伺服状态，先下使能
        if initial_state != 0:
            logger.warning(f"当前状态不是下伺服(0)，先下使能...")
            self.robot.clear_set()
            self.robot.set_state(arm=arm, state=0)
            self.robot.send_cmd()
            if not self.wait_for_state(arm, 0, timeout=3.0):
                logger.error(f"无法下使能 {arm} 臂")
                return False
            logger.info("✓ 已下使能")
            time.sleep(0.5)

        # 1. 切换到位置跟随模式
        logger.info("步骤 1: 切换到位置跟随模式...")
        self.robot.clear_set()
        self.robot.set_state(arm=arm, state=1)  # state=1 位置跟随
        self.robot.send_cmd()

        if not self.wait_for_state(arm, 1, timeout=5.0):
            sub_data = self.robot.subscribe(self.dcss)
            cur_state = sub_data['states'][arm_idx]['cur_state']
            logger.error(f"✗ 切换到位置跟随模式失败，当前状态: {cur_state}")
            return False

        logger.info("✓ 已进入位置跟随模式")
        time.sleep(0.3)

        # 2. 设置速度和加速度
        logger.info(f"步骤 2: 设置速度={vel_ratio}%, 加速度={acc_ratio}%...")
        self.robot.clear_set()
        self.robot.set_vel_acc(arm=arm, velRatio=vel_ratio, AccRatio=acc_ratio)
        self.robot.send_cmd()
        time.sleep(0.3)

        # 3. 发送目标位置
        logger.info("步骤 3: 发送目标位置...")
        self.robot.clear_set()
        self.robot.set_joint_cmd_pose(arm=arm, joints=home_joints)
        self.robot.send_cmd()

        # 4. 等待到达目标位置
        logger.info("步骤 4: 等待到达目标位置...")
        max_wait_time = 30.0  # 最多等待30秒
        start_time = time.time()
        reached = False

        while (time.time() - start_time) < max_wait_time:
            sub_data = self.robot.subscribe(self.dcss)
            current_joints = sub_data['outputs'][arm_idx]['fb_joint_pos']

            # 计算位置误差
            errors = [abs(current_joints[i] - home_joints[i]) for i in range(7)]
            max_error = max(errors)

            logger.info(f"  最大误差: {max_error:.3f}度")

            # 如果误差小于0.5度，认为到达
            if max_error < 0.5:
                reached = True
                break

            time.sleep(0.5)

        if reached:
            logger.info(f"✓ {arm} 臂已到达home位置")
            final_joints = self.robot.subscribe(self.dcss)['outputs'][arm_idx]['fb_joint_pos']
            logger.info(f"最终关节位置: {[round(j, 2) for j in final_joints]}")
            return True
        else:
            logger.warning(f"⚠ {arm} 臂未能在{max_wait_time}秒内到达home位置")
            final_joints = self.robot.subscribe(self.dcss)['outputs'][arm_idx]['fb_joint_pos']
            logger.warning(f"最终关节位置: {[round(j, 2) for j in final_joints]}")
            return False

    def move_to_joints_in_impedance_mode(self, arm='B', target_joints=None, K=None, D=None, timeout=30.0):
        """在关节阻抗模式下移动到指定角度（柔顺运动）

        Args:
            arm: 'A' 或 'B'
            target_joints: 目标关节角度 [7个值]
            K: 刚度参数 [7个值]，默认 [2,2,2,1,1,1,1]
            D: 阻尼参数 [7个值]，默认 [0.5,0.5,0.5,0.3,0.3,0.3,0.3]
            timeout: 超时时间（秒）
        """
        if not self.connected:
            logger.error("未连接到机器人")
            return False

        if target_joints is None:
            logger.error("必须提供目标关节角度")
            return False

        # 默认阻抗参数
        if K is None:
            K = [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]
        if D is None:
            D = [0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3]

        logger.info(f"\n{'='*50}")
        logger.info(f"在关节阻抗模式下移动 {arm} 臂到目标位置...")
        logger.info(f"{'='*50}")
        logger.info(f"目标关节位置: {[round(j, 2) for j in target_joints]}")
        logger.info(f"刚度 K: {K}")
        logger.info(f"阻尼 D: {D}")

        arm_idx = 0 if arm == 'A' else 1

        # 获取当前位置
        sub_data = self.robot.subscribe(self.dcss)
        current_joints = sub_data['outputs'][arm_idx]['fb_joint_pos']
        logger.info(f"当前关节位置: {[round(j, 2) for j in current_joints]}")

        # 先进入关节阻抗模式
        if not self.enter_joint_impedance_mode(arm=arm, K=K, D=D):
            logger.error("无法进入关节阻抗模式")
            return False

        # 在扭矩模式下发送目标关节位置
        logger.info("发送目标关节位置...")
        self.robot.clear_set()
        self.robot.set_joint_cmd_pose(arm=arm, joints=target_joints)
        self.robot.send_cmd()

        # 等待到达目标位置
        logger.info(f"等待到达目标位置（超时{timeout}秒）...")
        start_time = time.time()
        reached = False

        while (time.time() - start_time) < timeout:
            sub_data = self.robot.subscribe(self.dcss)
            current_joints = sub_data['outputs'][arm_idx]['fb_joint_pos']

            # 计算位置误差
            errors = [abs(current_joints[i] - target_joints[i]) for i in range(7)]
            max_error = max(errors)

            logger.info(f"  当前位置: {[round(j, 2) for j in current_joints]} | 最大误差: {max_error:.3f}度")

            # 如果误差小于1度，认为到达（阻抗模式下允许更大误差）
            if max_error < 1.0:
                reached = True
                break

            time.sleep(0.5)

        if reached:
            logger.info(f"✓ {arm} 臂已到达目标位置")
            final_joints = self.robot.subscribe(self.dcss)['outputs'][arm_idx]['fb_joint_pos']
            logger.info(f"最终关节位置: {[round(j, 2) for j in final_joints]}")
            logger.info("⚠ 提示: 机械臂仍处于关节阻抗模式，可以手动拖动调整位置")
            return True
        else:
            logger.warning(f"⚠ {arm} 臂未能在{timeout}秒内到达目标位置")
            final_joints = self.robot.subscribe(self.dcss)['outputs'][arm_idx]['fb_joint_pos']
            logger.warning(f"最终关节位置: {[round(j, 2) for j in final_joints]}")
            logger.info("⚠ 提示: 机械臂仍处于关节阻抗模式，可以手动拖动调整位置")
            return False

    def move_joints_slowly(self, arm='B', target_joints=None, vel_ratio=5, acc_ratio=5):
        """慢速移动关节（用于脱离限位）

        Args:
            arm: 'A' 或 'B'
            target_joints: 目标关节角度 [7个值]
            vel_ratio: 速度百分比 (1-100)，默认5%很慢
            acc_ratio: 加速度百分比 (1-100)，默认5%
        """
        if not self.connected:
            logger.error("未连接到机器人")
            return False

        if target_joints is None:
            logger.error("必须提供目标关节角度")
            return False

        logger.info(f"\n{'='*50}")
        logger.info(f"慢速移动 {arm} 臂关节（脱离限位用）...")
        logger.info(f"{'='*50}")
        logger.info(f"目标关节位置: {target_joints}")
        logger.info(f"速度: {vel_ratio}%, 加速度: {acc_ratio}%")

        arm_idx = 0 if arm == 'A' else 1

        # 获取当前位置
        sub_data = self.robot.subscribe(self.dcss)
        current_joints = sub_data['outputs'][arm_idx]['fb_joint_pos']
        logger.info(f"当前关节位置: {[round(j, 2) for j in current_joints]}")

        # 切换到位置模式
        initial_state = sub_data['states'][arm_idx]['cur_state']
        if initial_state != 1:
            logger.info("切换到位置跟随模式...")
            self.robot.clear_set()
            self.robot.set_state(arm=arm, state=1)
            self.robot.send_cmd()

            if not self.wait_for_state(arm, 1, timeout=5.0):
                logger.error("无法切换到位置跟随模式")
                return False
            time.sleep(0.3)

        # 设置低速
        self.robot.clear_set()
        self.robot.set_vel_acc(arm=arm, velRatio=vel_ratio, AccRatio=acc_ratio)
        self.robot.send_cmd()
        time.sleep(0.3)

        # 发送目标
        self.robot.clear_set()
        self.robot.set_joint_cmd_pose(arm=arm, joints=target_joints)
        self.robot.send_cmd()

        logger.info("✓ 已发送慢速移动指令")
        logger.info("⚠ 请观察机械臂运动，如有异常请立即急停！")

        return True

    def disconnect(self):
        """断开连接"""
        if self.connected:
            logger.info("断开机器人连接...")

            # 禁用夹爪电机
            if self._gripper is not None:
                for motor in (self._motor_left, self._motor_right):
                    try:
                        self._gripper.disable(motor)
                    except Exception:
                        pass

            self.robot.release_robot()
            self.connected = False
            logger.info("✓ 已断开连接")


def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  机械臂清错和阻抗/拖动模式控制程序")
    print("="*60)

    # 创建控制器
    controller = ArmController(robot_ip='192.168.10.190')

    # 连接机器人
    if not controller.connect():
        print("\n连接失败，程序退出")
        return

    try:
        while True:
            print("\n" + "-"*60)
            print("0609")
            print("请选择操作:")
            print("  1. 检查并清除A臂错误")
            print("  2. 检查并清除B臂错误")
            print("  3. A臂进入关节阻抗模式")
            print("  4. B臂进入关节阻抗模式")
            print("  5. A臂进入笛卡尔阻抗模式")
            print("  6. B臂进入笛卡尔阻抗模式")
            print("  7. A臂进入拖动模式")
            print("  8. B臂进入拖动模式")
            print("  9. 查看A臂当前关节位置")
            print(" 10. 查看B臂当前关节位置")
            print(" 11. 退出拖动模式并下使能")
            print(" 12. A臂回到home位置（关节复位）")
            print(" 13. B臂回到home位置（关节复位）")
            print(" 14. A臂慢速移动（脱离限位）")
            print(" 15. B臂慢速移动（脱离限位）")
            print(" 16. A臂在阻抗模式下移动到指定角度")
            print(" 17. B臂在阻抗模式下移动到指定角度")
            print(" 18. 同时下使能A臂和B臂")
            print(" 19. 初始化双夹爪")
            print(" 20. 下使能双夹爪")
            print(" 21. 移动左夹爪(A臂)到指定角度")
            print(" 22. 移动右夹爪(B臂)到指定角度")
            print("  0. 退出程序")
            print("-"*60)

            choice = input("请输入选项 (0-22): ").strip()

            if choice == '0':
                break
            elif choice == '1':
                controller.check_and_clear_errors('A')
            elif choice == '2':
                controller.check_and_clear_errors('B')
            elif choice == '3':
                controller.enter_joint_impedance_mode('A')
            elif choice == '4':
                controller.enter_joint_impedance_mode('B')
            elif choice == '5':
                controller.enter_cart_impedance_mode('A')
            elif choice == '6':
                controller.enter_cart_impedance_mode('B')
            elif choice == '7':
                print("\n拖动类型:")
                print("  1. 关节空间拖动")
                print("  2. 笛卡尔X方向拖动")
                print("  3. 笛卡尔Y方向拖动")
                print("  4. 笛卡尔Z方向拖动")
                print("  5. 笛卡尔旋转拖动")
                drag_type = int(input("请选择拖动类型 (1-5): ").strip())
                controller.enter_drag_mode('A', drag_type)
            elif choice == '8':
                print("\n拖动类型:")
                print("  1. 关节空间拖动")
                print("  2. 笛卡尔X方向拖动")
                print("  3. 笛卡尔Y方向拖动")
                print("  4. 笛卡尔Z方向拖动")
                print("  5. 笛卡尔旋转拖动")
                drag_type = int(input("请选择拖动类型 (1-5): ").strip())
                controller.enter_drag_mode('B', drag_type)
            elif choice == '9':
                pos = controller.get_current_joint_positions('A')
                if pos:
                    logger.info(f"A臂当前关节位置: {[round(p, 2) for p in pos]}")
            elif choice == '10':
                pos = controller.get_current_joint_positions('B')
                if pos:
                    logger.info(f"B臂当前关节位置: {[round(p, 2) for p in pos]}")
            elif choice == '11':
                arm = input("退出哪个臂? (A/B): ").strip().upper()
                if arm in ['A', 'B']:
                    controller.exit_drag_mode(arm)
                    controller.disable_arm(arm)
            elif choice == '12':
                print("\n选择home位置类型:")
                print("  1. 标准home位置（左臂默认镜像姿态）")
                print("  2. 自定义位置")
                home_choice = input("请选择 (1-2): ").strip()

                if home_choice == '1':
                    vel = int(input("速度百分比 (1-100, 推荐10): ") or "10")
                    controller.move_to_home_position('A', vel_ratio=vel, acc_ratio=vel)
                elif home_choice == '2':
                    print("输入7个关节角度，用逗号或空格分隔:")
                    joints_str = input("关节角度: ").strip()
                    joints = [float(x.strip()) for x in joints_str.replace(',', ' ').split()]
                    if len(joints) == 7:
                        vel = int(input("速度百分比 (1-100, 推荐10): ") or "10")
                        controller.move_to_home_position('A', home_joints=joints, vel_ratio=vel, acc_ratio=vel)
                    else:
                        print(f"错误: 需要7个关节角度，你输入了{len(joints)}个")
            elif choice == '13':
                print("\n选择home位置类型:")
                print("  1. 标准home位置（右臂默认镜像姿态）")
                print("  2. 自定义位置")
                home_choice = input("请选择 (1-2): ").strip()

                if home_choice == '1':
                    vel = int(input("速度百分比 (1-100, 推荐10): ") or "10")
                    controller.move_to_home_position('B', vel_ratio=vel, acc_ratio=vel)
                elif home_choice == '2':
                    print("输入7个关节角度，用逗号或空格分隔:")
                    joints_str = input("关节角度: ").strip()
                    joints = [float(x.strip()) for x in joints_str.replace(',', ' ').split()]
                    if len(joints) == 7:
                        vel = int(input("速度百分比 (1-100, 推荐10): ") or "10")
                        controller.move_to_home_position('B', home_joints=joints, vel_ratio=vel, acc_ratio=vel)
                    else:
                        print(f"错误: 需要7个关节角度，你输入了{len(joints)}个")
            elif choice == '14':
                pos = controller.get_current_joint_positions('A')
                if pos:
                    print(f"\nA臂当前关节位置: {[round(p, 2) for p in pos]}")
                    print("输入目标关节角度（7个值，用逗号或空格分隔）:")
                    joints_str = input("目标角度: ").strip()
                    joints = [float(x.strip()) for x in joints_str.replace(',', ' ').split()]
                    if len(joints) == 7:
                        vel = int(input("速度百分比 (1-100, 推荐5): ") or "5")
                        controller.move_joints_slowly('A', target_joints=joints, vel_ratio=vel, acc_ratio=vel)
                    else:
                        print(f"错误: 需要7个关节角度，你输入了{len(joints)}个")
            elif choice == '15':
                pos = controller.get_current_joint_positions('B')
                if pos:
                    print(f"\nB臂当前关节位置: {[round(p, 2) for p in pos]}")
                    print("输入目标关节角度（7个值，用逗号或空格分隔）:")
                    joints_str = input("目标角度: ").strip()
                    joints = [float(x.strip()) for x in joints_str.replace(',', ' ').split()]
                    if len(joints) == 7:
                        vel = int(input("速度百分比 (1-100, 推荐5): ") or "5")
                        controller.move_joints_slowly('B', target_joints=joints, vel_ratio=vel, acc_ratio=vel)
                    else:
                        print(f"错误: 需要7个关节角度，你输入了{len(joints)}个")
            elif choice == '16':
                pos = controller.get_current_joint_positions('A')
                if pos:
                    print(f"\nA臂当前关节位置: {[round(p, 2) for p in pos]}")
                    print("输入目标关节角度（7个值，用逗号或空格分隔）:")
                    joints_str = input("目标角度: ").strip()
                    joints = [float(x.strip()) for x in joints_str.replace(',', ' ').split()]
                    if len(joints) == 7:
                        print("\n是否使用自定义阻抗参数? (y/n, 默认n)")
                        use_custom = input().strip().lower()
                        if use_custom == 'y':
                            print("输入刚度K（7个值，用逗号或空格分隔，推荐[2,2,2,1,1,1,1]）:")
                            k_str = input("K: ").strip()
                            K = [float(x.strip()) for x in k_str.replace(',', ' ').split()]
                            print("输入阻尼D（7个值，用逗号或空格分隔，推荐[0.5,0.5,0.5,0.3,0.3,0.3,0.3]）:")
                            d_str = input("D: ").strip()
                            D = [float(x.strip()) for x in d_str.replace(',', ' ').split()]
                            if len(K) == 7 and len(D) == 7:
                                controller.move_to_joints_in_impedance_mode('A', target_joints=joints, K=K, D=D)
                            else:
                                print(f"错误: K和D都需要7个值")
                        else:
                            controller.move_to_joints_in_impedance_mode('A', target_joints=joints)
                    else:
                        print(f"错误: 需要7个关节角度，你输入了{len(joints)}个")
            elif choice == '17':
                pos = controller.get_current_joint_positions('B')
                if pos:
                    print(f"\nB臂当前关节位置: {[round(p, 2) for p in pos]}")
                    print("输入目标关节角度（7个值，用逗号或空格分隔）:")
                    joints_str = input("目标角度: ").strip()
                    joints = [float(x.strip()) for x in joints_str.replace(',', ' ').split()]
                    if len(joints) == 7:
                        print("\n是否使用自定义阻抗参数? (y/n, 默认n)")
                        use_custom = input().strip().lower()
                        if use_custom == 'y':
                            print("输入刚度K（7个值，用逗号或空格分隔，推荐[2,2,2,1,1,1,1]）:")
                            k_str = input("K: ").strip()
                            K = [float(x.strip()) for x in k_str.replace(',', ' ').split()]
                            print("输入阻尼D（7个值，用逗号或空格分隔，推荐[0.5,0.5,0.5,0.3,0.3,0.3,0.3]）:")
                            d_str = input("D: ").strip()
                            D = [float(x.strip()) for x in d_str.replace(',', ' ').split()]
                            if len(K) == 7 and len(D) == 7:
                                controller.move_to_joints_in_impedance_mode('B', target_joints=joints, K=K, D=D)
                            else:
                                print(f"错误: K和D都需要7个值")
                        else:
                            controller.move_to_joints_in_impedance_mode('B', target_joints=joints)
                    else:
                        print(f"错误: 需要7个关节角度，你输入了{len(joints)}个")
            elif choice == '18':
                controller.disable_both_arms()
            elif choice == '19':
                controller.init_gripper()
            elif choice == '20':
                controller.disable_gripper()
            elif choice == '21':
                if controller._gripper is None:
                    print("夹爪未初始化，请先选择选项19初始化夹爪")
                else:
                    print("\n左夹爪(A臂)控制")
                    print("输入目标角度（弧度，推荐范围: 0.0 到 1.5）:")
                    try:
                        target = float(input("目标角度: ").strip())
                        stiffness = float(input("刚度 (推荐8.0): ") or "8.0")
                        damping = float(input("阻尼 (推荐0.20): ") or "0.20")
                        controller.move_gripper_to_position('left', target, stiffness, damping)
                    except ValueError:
                        print("输入格式错误，请输入数字")
            elif choice == '22':
                if controller._gripper is None:
                    print("夹爪未初始化，请先选择选项19初始化夹爪")
                else:
                    print("\n右夹爪(B臂)控制")
                    print("输入目标角度（弧度，推荐范围: 0.0 到 1.5）:")
                    try:
                        target = float(input("目标角度: ").strip())
                        stiffness = float(input("刚度 (推荐8.0): ") or "8.0")
                        damping = float(input("阻尼 (推荐0.20): ") or "0.20")
                        controller.move_gripper_to_position('right', target, stiffness, damping)
                    except ValueError:
                        print("输入格式错误，请输入数字")
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
