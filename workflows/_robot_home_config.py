#!/usr/bin/env python3
"""_robot_home_config.py — 机械臂 home 位置的中心配置

所有需要使用 home 位置的代码都应该从这里导入，而不是自己定义。
这样保证了整个项目中 home 位置的一致性。

更新 home 位置：
    1. 运行 workflows/arm_control_http.py 连接机械臂
    2. 选择选项 1 查看当前位置
    3. 把当前位置更新到下面的 HOME_LEFT_ARM
    4. 旧的 home 位置会自动作为注释保留
"""

# 左臂（A 臂）标准 home 位置（单位：度）
# 更新时间：初始定义
# HOME_LEFT_ARM = [
#     66.04866790771484,
#     -18.997726440429688,
#     -80.62322998046875,
#     -84.70333862304688,
#     -47.016021728515625,
#     31.47335433959961,
#     -40.16086959838867,
# ]
HOME_LEFT_ARM = [
97.42, -62.95, -62.8, -114.38, -21.22, 7.35, 31.64
]
# 双臂镜像规则：索引 0/2/4/6 取反，索引 1/3/5 保持一致
_MIRROR_INDICES = (0, 2, 4, 6)


def get_home_right_arm():
    """根据左臂 home 位置和镜像规则生成右臂 home 位置"""
    home_right = list(HOME_LEFT_ARM)
    for i in _MIRROR_INDICES:
        home_right[i] = -home_right[i]
    return home_right


def get_home_position(arm: str = None):
    """获取 home 位置

    Args:
        arm: 'A' 或 'left' 返回左臂，'B' 或 'right' 返回右臂，
             None 返回 (左臂, 右臂) 元组

    Returns:
        list 或 tuple: home 位置（度数）
    """
    if arm in ('A', 'left'):
        return list(HOME_LEFT_ARM)
    elif arm in ('B', 'right'):
        return get_home_right_arm()
    elif arm is None:
        return (list(HOME_LEFT_ARM), get_home_right_arm())
    else:
        raise ValueError(f"arm 必须是 'A'/'left' 或 'B'/'right'，得到: {arm}")


def get_home_action_16joints():
    """获取完整的 16 关节 home 动作（14个臂关节 + 2个夹爪）

    Returns:
        list: [左臂7关节, 右臂7关节, 左夹爪, 右夹爪]
    """
    left, right = get_home_position()
    return left + right + [0.0, 0.0]


# 历史 home 位置（作为备份参考）
# --------------------------------------------------
# 初始定义 (日期未知):
#   HOME_LEFT_ARM = [66.04866790771484, -18.997726440429688, -80.62322998046875,
#                    -84.70333862304688, -47.016021728515625, 31.47335433959961, -40.16086959838867]
