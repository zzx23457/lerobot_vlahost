# frame_serial 验证失败问题诊断报告

**日期**: 2026-06-18  
**问题**: `arm_clear_and_impedance.py` 执行失败，但运行 `test_connect_simple.py` 后可以成功

## 问题现象

- 直接运行 `arm_clear_and_impedance.py` 时，连接验证失败（frame_serial 为0）
- 先运行 `test_connect_simple.py`，再运行 `arm_clear_and_impedance.py` 则成功
- `test_connect_simple.py` 似乎是 `arm_clear_and_impedance.py` 成功执行的前驱步骤

## 根本原因

### test_connect_simple.py 的做法（正确）

在验证连接时，主动**发送命令激活数据流**：

```python
# 第49-52行
print("\n3. 尝试发送命令激活数据流...")
robot.clear_set()
robot.log_switch('1')
robot.send_cmd()
time.sleep(0.5)
```

然后才检查 frame_serial 是否更新（第56-65行）。

### arm_clear_and_impedance.py 的问题

在 `connect()` 方法中（第52-65行），**直接循环检查 frame_serial，但没有先发送命令激活数据流**：

```python
# 验证连接
time.sleep(1.0)
motion_tag = 0
frame_update = None

for i in range(5):
    sub_data = self.robot.subscribe(self.dcss)
    frame_serial = sub_data['outputs'][0]['frame_serial']
    
    if frame_serial != 0 and frame_update != frame_serial:
        motion_tag += 1
        frame_update = frame_serial
    time.sleep(0.2)
```

如果机器人控制器的数据流未被激活，frame_serial 会一直为0，导致 `motion_tag` 为0，连接验证失败。

## 为什么 test_connect_simple.py 是"前驱步骤"

运行 `test_connect_simple.py` 后：
1. 通过 `robot.clear_set()` + `robot.log_switch('1')` + `robot.send_cmd()` 激活了数据流
2. 机器人控制器的共享内存开始正常更新
3. frame_serial 开始递增
4. 后续运行 `arm_clear_and_impedance.py` 时，数据流已经激活，能够通过验证

## 解决方案

修改 `arm_clear_and_impedance.py` 的 `connect()` 方法，在检查 frame_serial **之前**先激活数据流：

```python
def connect(self):
    """连接机器人"""
    logger.info(f"正在连接机器人 {self.robot_ip}...")
    init = self.robot.connect(self.robot_ip)
    
    if init == 0:
        logger.error('连接失败! 端口可能被占用')
        return False
    
    # 验证连接
    time.sleep(1.0)
    
    # ✨ 新增：发送命令激活数据流
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
```

## 修改位置

- 文件：`arm_clear_and_impedance.py`
- 方法：`ArmController.connect()`
- 行号：第52-76行
- 修改：在第53行 `time.sleep(1.0)` 之后，插入激活数据流的代码

## 测试建议

修改后应该测试：
1. 直接运行修改后的脚本（不先运行 test_connect_simple.py）
2. 验证连接是否成功
3. 验证后续的阻抗模式设置功能是否正常
