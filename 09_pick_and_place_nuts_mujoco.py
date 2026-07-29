#!/usr/bin/env python3
"""
09_pick_and_place_nuts_mujoco.py
---------------------------------
基于 gen3_with_gripper_and_nuts.xml（Kinova Gen3 机械臂 + Robotiq 2F-85 夹爪 +
D435i 相机 + 工作台 + square_nut / square_nut_2 / round_nut 三个螺母）的
"抓取-放置"（pick & place）运动学可视化演示脚本。

── 起始位姿 ────────────────────────────────────────────────────────────
机械臂的起始关节角直接取自 pose_estimation_data_3nuts/poses/frame_000000.json
中保存的 joint_angles_rad（即 05_collect_pose_estimation_data_mujoco.py /
07_verify_pose_estimation.py / 08_compute_nut_pose_in_base.py 这一整条
标定验证流水线里，采集该帧数据时机械臂实际所处的姿态）。

── 抓取目标位姿：为何不直接复用 output_3nuts 里 FoundationPose 的估计结果 ──
output_3nuts/nut_pose_in_base_000000.json 里 08 脚本算出的 T_grasp_base_est
是针对 pose_estimation_data_3nuts 对应的**原始**三螺母场景
（gen3_with_two_nuts_and_round_nut.xml，螺母世界坐标 y≈0.65，用于远距离
相机大视野采集）计算的；实测该场景下螺母到机械臂基座的距离约 1.23~1.28 m，
已超出 Kinova Gen3 7 轴臂约 0.9 m 的最大可达范围——如果直接照搬这个抓取目标，
逆运动学根本无法收敛（不是标定或代码 bug，是物理上"够不着"）。

因此 gen3_with_gripper_and_nuts.xml 已把 3 个螺母、工作台、放置目标标记整体
沿 -Y 方向平移到机械臂实际可达范围内（螺母 y≈0.15、放置目标 y≈-0.05，参见该
文件中 "table" body 上方的说明注释），螺母间 / 抓取区与放置区间的相对布局
保持不变。因为螺母的世界坐标发生了变化，本脚本改为：直接读取当前场景中
square_nut / square_nut_2 / round_nut 三个 body 的仿真真值位姿
T_nut_base（与 08 脚本 compute_T_nut_base_gt() 用的是完全相同的一套坐标
变换：T_nut_base = inv(T_base_world) @ T_nut_world），再用与 08 脚本一致的
公式 T_grasp_base = T_nut_base @ T_grasp_nut（T_grasp_nut 仍来自
nut_grasp_pose.json / nut_grasp_pose_round.json，与之前完全一致，未改变
任何抓取几何定义）算出每个螺母的抓取目标。也就是说：本脚本只是把"从图像
估计 T_nut_base"换成了"从仿真真值直接读取 T_nut_base"（对于一个纯运动学
可视化演示脚本，这样做既避免了不必要的 GPU 推理开销，也保证了目标点必然
落在可达范围内），抓取偏移量的定义、变换公式与 08/07 脚本完全一致。

── 放置目标位姿 ───────────────────────────────────────────────────────
放置位置由本脚本直接定义（未做标定/检测，纯人为指定的目标点）：
把每个螺母从初始摆放行（世界坐标 y≈0.15）沿 -Y 方向移动到 y≈-0.05 的
放置目标行，x 坐标与该螺母的初始位置保持一致（对应 gen3_with_gripper_and_nuts.xml
中新增的三个圆形 place_marker_* 半透明色块标记），抓取时的姿态（TCP 相对
物体的朝向）在放置时保持不变，只改变位置。

── TCP-法兰标定 ─────────────────────────────────────────────────────────
tcp_flange.json 已更新为由 gen3_with_gripper_and_nuts.xml 运动学树直接算出
的真实变换（不再是占位单位阵），本脚本用它算出机械臂法兰应到达的目标位姿
T_flange_base_est = T_grasp_base_est @ inv(T_tcp_flange)，公式与
08_compute_nut_pose_in_base.py 完全一致。

── 场景补充：工作台 ────────────────────────────────────────────────────
gen3_with_gripper_and_nuts.xml 中已新增 "table" 静态实体（棕色木纹箱体 +
四条桌腿），台面高度与螺母摆放高度 z=0.85 匹配，同时覆盖初始摆放区
（y≈0.65）与放置目标区（y≈0.45）。

── 运动学求解方法 ───────────────────────────────────────────────────────
机械臂 7 个关节角由阻尼最小二乘（Damped Least Squares，DLS）雅可比逆运动学
（IK）求解：给定夹爪 TCP（site "grip_site"，与 tcp_flange.json 中定义的
"eef" body 原点重合）在世界坐标系下的目标位置 + 目标姿态，用
mujoco.mj_jacSite 取该 site 的平移/旋转雅可比，迭代更新 7 个关节角直至
位置误差 < 1 mm 且姿态误差 < 0.5°。相邻两个关键路点之间的关节角在关节
空间中线性插值，得到平滑运动轨迹（本脚本按"运动学示教"方式直接改写
data.qpos + mujoco.mj_forward 渲染每一帧，不做力矩/动力学仿真，因此螺母、
桌子等固定 body 不会受重力等物理效应影响，只有机械臂和夹爪的关节角在动画
过程中变化——这与 08/07 脚本中"仅用于姿态验证的纯运动学计算"是同一思路的
可视化延伸）。

── 夹爪开合 ────────────────────────────────────────────────────────────
Robotiq 2F-85 的 6 个夹爪关节（finger_joint 等）同样直接用 qpos 插值实现
张开(全部置 0)/闭合(约 0.68 rad，对应约 60% 行程，足以夹住螺母把手同时留有
安全裕量)动画，不经过 tendon/actuator 的物理约束求解（原因同上：本脚本是
纯运动学可视化，不做动力学仿真）。

── 可视化与视频保存 ─────────────────────────────────────────────────────
默认通过 mujoco.viewer.launch_passive 打开交互式查看器窗口实时显示整个
抓取-放置过程（--no_viewer 可关闭，适合无显示环境下只保存视频）；
同时可用 --save_video 指定 mp4 输出路径，用独立的离屏 Renderer 逐帧渲染
并通过 OpenCV 写入视频文件（帧率由 --fps 控制）。

用法示例
--------
  # 交互式查看 + 保存视频
  python 09_pick_and_place_nuts_mujoco.py \\
      --xml gen3_with_gripper_and_nuts.xml \\
      --start_pose_file pose_estimation_data_3nuts/poses/frame_000000.json \\
      --nut_pose_file output_3nuts/nut_pose_in_base_000000.json \\
      --tcp_flange_file tcp_flange.json \\
      --save_video output_3nuts/pick_and_place.mp4

  # 仅保存视频，不弹出交互窗口（无显示环境 / 服务器上运行）
  python 09_pick_and_place_nuts_mujoco.py --no_viewer \\
      --save_video output_3nuts/pick_and_place.mp4
"""

import argparse
import json
import os
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# ─────────────────────────────────────────────────────────────────────────────
ARM_JOINT_NAMES = ["joint_1", "joint_2", "joint_3",
                   "joint_4", "joint_5", "joint_6", "joint_7"]
# Robotiq 2F-85 六个关节：仅 finger_joint / right_outer_knuckle_joint 由执行器直接
# 驱动，其余四个通过 tendon 耦合；本脚本纯运动学演示，直接对全部 6 个关节做
# qpos 插值（近似同步张合，不经过 tendon 约束求解，视觉效果一致）。
GRIPPER_JOINT_NAMES = [
    "finger_joint", "left_inner_finger_joint", "left_inner_knuckle_joint",
    "right_outer_knuckle_joint", "right_inner_finger_joint", "right_inner_knuckle_joint",
]
GRIPPER_OPEN_QPOS = {
    "finger_joint": 0.0, "left_inner_finger_joint": 0.0, "left_inner_knuckle_joint": 0.0,
    "right_outer_knuckle_joint": 0.0, "right_inner_finger_joint": 0.0, "right_inner_knuckle_joint": 0.0,
}
GRIPPER_CLOSE_QPOS = {
    "finger_joint": 0.68, "left_inner_finger_joint": -0.60, "left_inner_knuckle_joint": 0.68,
    "right_outer_knuckle_joint": 0.68, "right_inner_finger_joint": -0.60, "right_inner_knuckle_joint": 0.68,
}

GRIP_SITE = "grip_site"
BASE_BODY_NAME = "base_link"

# 抓取/放置动作中，垂直提升高度（世界坐标系 Z 方向，米）
LIFT_HEIGHT = 0.15
# 抓取前"预抓取点"沿夹爪接近方向（grasp 姿态局部 +Z）的后退距离（米）
PREGRASP_BACKOFF = 0.12

# 螺母名 -> 世界坐标系放置目标 Y 坐标（基坐标系/世界坐标系下 X/Z 与抓取点保持
# 一致，只改 Y，见 gen3_with_gripper_and_nuts.xml 中 place_marker_* 的位置）。
PLACE_WORLD_Y = -0.05

# 螺母 body 名 -> 抓取位姿定义文件（T_grasp_nut：夹爪 TCP 相对螺母局部坐标系
# 的固定抓取变换），与 08_compute_nut_pose_in_base.py 使用的是同一批文件。
NUT_BODY_NAMES = ["square_nut", "square_nut_2", "round_nut"]
DEFAULT_GRASP_POSE_FILES = {
    "square_nut": "nut_grasp_pose.json",
    "square_nut_2": "nut_grasp_pose.json",
    "round_nut": "nut_grasp_pose_round.json",
}


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def make_T(pos, mat):
    T = np.eye(4)
    T[:3, :3] = np.array(mat).reshape(3, 3)
    T[:3, 3] = np.array(pos)
    return T


def get_body_T_world(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"找不到 body: {body_name}")
    return make_T(data.xpos[body_id], data.xmat[body_id])


def get_nut_freejoint_qpos_adr(model, nut_name):
    """获取螺母自由关节（<freejoint name="{nut_name}_freejoint"/>，见
    gen3_with_gripper_and_nuts.xml）在 qpos 数组中的起始下标（7 个连续值：
    3 个平移 + 4 元数 wxyz 旋转）。"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{nut_name}_freejoint")
    if jid == -1:
        raise ValueError(f"找不到自由关节: {nut_name}_freejoint（请确认场景 xml "
                          f"中该螺母 body 下是否已添加 <freejoint>）")
    return model.jnt_qposadr[jid]


def set_nut_world_pose(model, data, nut_name, T_world):
    """把螺母（带 <freejoint>）的自由关节 qpos 直接设为给定的世界位姿 T_world
    （4x4 齐次变换）。用于 09 脚本里"运动学抓取"：在夹爪闭合到放开的整个
    搬运过程中，每一帧都把螺母的世界位姿重新计算为
        T_nut_world = T_grip_world_current @ T_nut_grip_at_grasp_time
    从而让螺母看起来像被夹爪"刚性抓住"一起移动（并非真实物理接触/摩擦力
    约束，而是直接写位姿的运动学近似，对可视化演示已经足够）。
    """
    adr = get_nut_freejoint_qpos_adr(model, nut_name)
    pos = T_world[:3, 3]
    quat_xyzw = Rotation.from_matrix(T_world[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    data.qpos[adr:adr + 3] = pos
    data.qpos[adr + 3:adr + 7] = quat_wxyz


def get_site_T_world(model, data, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"找不到 site: {site_name}")
    return make_T(data.site_xpos[site_id], data.site_xmat[site_id])


def set_joint_qpos(model, data, joint_name, value):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    addr = model.jnt_qposadr[jid]
    data.qpos[addr] = value


def get_joint_qpos(model, data, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    addr = model.jnt_qposadr[jid]
    return data.qpos[addr]


def set_arm_qpos(model, data, q7):
    for name, q in zip(ARM_JOINT_NAMES, q7):
        set_joint_qpos(model, data, name, q)


def get_arm_qpos(model, data):
    return np.array([get_joint_qpos(model, data, n) for n in ARM_JOINT_NAMES])


def set_gripper_qpos(model, data, ratio):
    """ratio=0 -> 完全张开，ratio=1 -> 完全闭合（GRIPPER_CLOSE_QPOS）。"""
    for name in GRIPPER_JOINT_NAMES:
        v = GRIPPER_OPEN_QPOS[name] + ratio * (GRIPPER_CLOSE_QPOS[name] - GRIPPER_OPEN_QPOS[name])
        set_joint_qpos(model, data, name, v)


def rotation_log(R):
    """旋转矩阵 -> 轴角向量（rotvec，用于姿态误差 / 姿态插值）。"""
    return Rotation.from_matrix(R).as_rotvec()


def solve_ik(model, data, target_pos, target_mat, q0,
             max_iter=300, pos_tol=1e-3, rot_tol_deg=0.5, damping=1e-3, step_clip=0.2):
    """阻尼最小二乘（DLS）雅可比逆运动学，求解 7 个关节角使 site "grip_site"
    的世界位姿逼近 (target_pos, target_mat)。q0 为迭代初值（热启动，保证解的
    连续性，避免相邻路点之间关节角发生大跳变）。"""
    set_arm_qpos(model, data, q0)
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    arm_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES]
    dof_idx = [model.jnt_dofadr[j] for j in arm_jids]

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(max_iter):
        pos = data.site_xpos[site_id].copy()
        mat = data.site_xmat[site_id].reshape(3, 3).copy()

        pos_err = target_pos - pos
        rot_err = rotation_log(target_mat @ mat.T)
        err = np.concatenate([pos_err, rot_err])

        rot_err_deg = np.degrees(np.linalg.norm(rot_err))
        if np.linalg.norm(pos_err) < pos_tol and rot_err_deg < rot_tol_deg:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp, jacr])[:, dof_idx]  # (6, 7)

        JJt = J @ J.T + damping * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, err)
        dq = np.clip(dq, -step_clip, step_clip)

        q = get_arm_qpos(model, data)
        set_arm_qpos(model, data, q + dq)
        mujoco.mj_forward(model, data)

    return get_arm_qpos(model, data)


def slerp_rotmat(R0, R1, t):
    rv = rotation_log(R1 @ R0.T)
    R_delta = Rotation.from_rotvec(rv * t).as_matrix()
    return R_delta @ R0


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_start_qpos(start_pose_file):
    with open(start_pose_file, encoding="utf-8") as f:
        d = json.load(f)
    return np.array(d["joint_angles_rad"], dtype=float)


def load_tcp_flange(tcp_flange_file):
    with open(tcp_flange_file, encoding="utf-8") as f:
        d = json.load(f)
    return np.array(d["T_tcp_flange"])


def load_grasp_pose(grasp_pose_file):
    """加载预设抓取位姿文件，返回 T_grasp_nut（夹爪 TCP 相对目标物体局部坐标系），
    与 08_compute_nut_pose_in_base.py 中的同名函数完全一致。"""
    with open(grasp_pose_file, encoding="utf-8") as f:
        grasp = json.load(f)
    return np.array(grasp["T_grasp_nut"])


def compute_grasp_targets_from_ground_truth(model, data, here, grasp_pose_files=None):
    """直接从当前场景（gen3_with_gripper_and_nuts.xml）里 square_nut /
    square_nut_2 / round_nut 三个 body 的仿真真值位姿计算每个螺母的抓取目标
    T_grasp_base（基坐标系下），计算公式与 08_compute_nut_pose_in_base.py 的
    compute_T_nut_base_gt() + T_grasp_base = T_nut_base @ T_grasp_nut 完全一致，
    只是把"图像估计的 T_nut_base"换成了"仿真真值的 T_nut_base"（详见文件顶部
    docstring 中"为何不直接复用 output_3nuts 估计结果"一节的说明）。
    """
    grasp_pose_files = grasp_pose_files or {}
    T_base_world = get_body_T_world(model, data, BASE_BODY_NAME)
    targets = {}
    for nut_name in NUT_BODY_NAMES:
        T_nut_world = get_body_T_world(model, data, nut_name)
        T_nut_base = np.linalg.inv(T_base_world) @ T_nut_world

        grasp_file = grasp_pose_files.get(nut_name, DEFAULT_GRASP_POSE_FILES[nut_name])
        if not os.path.isabs(grasp_file):
            grasp_file = os.path.join(here, grasp_file)
        T_grasp_nut = load_grasp_pose(grasp_file)

        targets[nut_name] = T_nut_base @ T_grasp_nut
    return targets


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def build_waypoints(T_base_world, T_grasp_base):
    """给定某个螺母在基坐标系下的抓取目标 T_grasp_base，构造该螺母的完整
    抓取-放置路点序列（均为世界坐标系下的 (pos, rotmat) 对），依次为：
      pregrasp -> grasp -> lift -> preplace -> place -> retreat
    抓取/放置过程中姿态保持不变（只有 pregrasp/retreat 沿接近轴/竖直方向
    有位置偏移），对应"从上方垂直靠近抓取把手、原朝向放到新位置"的直观动作。
    """
    T_grasp_world = T_base_world @ T_grasp_base
    R = T_grasp_world[:3, :3]
    p_grasp = T_grasp_world[:3, 3]

    approach_axis_world = R[:, 2]  # 抓取坐标系局部 +Z（接近方向）在世界系下的方向
    p_pregrasp = p_grasp - PREGRASP_BACKOFF * approach_axis_world
    p_lift = p_grasp + np.array([0.0, 0.0, LIFT_HEIGHT])

    p_place = p_grasp.copy()
    p_place[1] = PLACE_WORLD_Y  # 只改世界 Y 坐标，X/Z 与抓取点保持一致
    p_preplace = p_place + np.array([0.0, 0.0, LIFT_HEIGHT])
    p_retreat = p_preplace.copy()

    waypoints = [
        # (名称, 目标位置, 目标姿态, 夹爪闭合比例, 该段是否需要"抓着螺母走")
        ("pregrasp", p_pregrasp, R, 0.0, False),
        ("grasp",    p_grasp,    R, 0.0, False),
        ("close",    p_grasp,    R, 1.0, True),   # 位置不变，仅闭合夹爪；从本段开始螺母跟随夹爪
        ("lift",     p_lift,     R, 1.0, True),
        ("preplace", p_preplace, R, 1.0, True),
        ("place",    p_place,    R, 1.0, True),
        ("open",     p_place,    R, 0.0, False),  # 位置不变，仅张开夹爪；松开螺母
        ("retreat",  p_retreat,  R, 0.0, False),
    ]
    return waypoints


def interpolate_and_run(model, data, nut_name, q_start, gripper_ratio_start,
                          waypoints, n_steps_per_segment, frame_cb):
    """在相邻路点之间做关节空间线性插值（夹爪开合比例也线性插值），逐帧
    调用 frame_cb(model, data) 渲染/记录。返回最终关节角与夹爪比例，供衔接
    下一个螺母的动作序列。

    ── 螺母"跟随抓取"的实现（运动学抓取，非真实物理约束）──────────────
    每个路点带有一个 attach 标志（见 build_waypoints）。当某一段的 attach
    从 False 变为 True 时（即将开始闭合夹爪并抬起），先记录此刻螺母相对
    夹爪 TCP（site "grip_site"）的相对位姿 T_nut_grip = inv(T_grip_world) @
    T_nut_world；此后只要 attach 仍为 True，每一帧都反过来用当前夹爪 TCP
    的世界位姿重新算出螺母应该在的世界位姿 T_nut_world = T_grip_world_now @
    T_nut_grip，并写入螺母 <freejoint> 的 qpos（见 set_nut_world_pose）。
    这样螺母就会像被夹爪"刚性抓住"一样跟随夹爪的位置和姿态变化一起移动，
    直到 attach 变回 False（对应"open"路点，松开夹爪）为止，螺母从此停留在
    松开时刻的世界位姿上（因为不再更新，退回到跟静态 body 完全一样的行为）。
    """
    q_prev = q_start.copy()
    ratio_prev = gripper_ratio_start
    attached = False
    T_nut_grip = None

    for name, p_target, R_target, ratio_target, attach_flag in waypoints:
        q_target = solve_ik(model, data, p_target, R_target, q_prev)

        if attach_flag and not attached:
            # 刚进入"抓着螺母走"的阶段：先把状态恢复到本段开始前（q_prev/
            # ratio_prev），读出此刻夹爪 TCP 与螺母的相对位姿，作为后续所有
            # 帧的固定"抓取偏移量"。
            set_arm_qpos(model, data, q_prev)
            set_gripper_qpos(model, data, ratio_prev)
            mujoco.mj_forward(model, data)
            T_grip_world = get_site_T_world(model, data, GRIP_SITE)
            T_nut_world = get_body_T_world(model, data, nut_name)
            T_nut_grip = np.linalg.inv(T_grip_world) @ T_nut_world
        attached = attach_flag

        for step in range(1, n_steps_per_segment + 1):
            t = step / n_steps_per_segment
            q_interp = q_prev + t * (q_target - q_prev)
            ratio_interp = ratio_prev + t * (ratio_target - ratio_prev)
            set_arm_qpos(model, data, q_interp)
            set_gripper_qpos(model, data, ratio_interp)
            mujoco.mj_forward(model, data)
            if attached:
                T_grip_world_now = get_site_T_world(model, data, GRIP_SITE)
                set_nut_world_pose(model, data, nut_name, T_grip_world_now @ T_nut_grip)
                mujoco.mj_forward(model, data)
            frame_cb(model, data)
        q_prev = q_target
        ratio_prev = ratio_target

    return q_prev, ratio_prev


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Gen3 + Robotiq 2F-85 抓取-放置运动学可视化演示（3 个螺母）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xml", default=os.path.join(here, "gen3_with_gripper_and_nuts.xml"))
    parser.add_argument("--start_pose_file",
                        default=os.path.join(here, "pose_estimation_data_3nuts", "poses", "frame_000000.json"),
                        help="机械臂起始关节角来源（joint_angles_rad 字段）")
    parser.add_argument("--grasp_pose_file", default=os.path.join(here, "nut_grasp_pose.json"),
                        help="方形螺母（square_nut/square_nut_2）共用的预设抓取位姿文件")
    parser.add_argument("--round_grasp_pose_file", default=os.path.join(here, "nut_grasp_pose_round.json"),
                        help="圆形螺母（round_nut）预设抓取位姿文件")
    parser.add_argument("--tcp_flange_file", default=os.path.join(here, "tcp_flange.json"))
    parser.add_argument("--nut_order", default="square_nut,square_nut_2,round_nut",
                        help="依次抓取/放置的螺母顺序（逗号分隔）")
    parser.add_argument("--steps_per_segment", type=int, default=60,
                        help="每两个相邻路点之间的插值帧数（越大动作越平滑越慢）")
    parser.add_argument("--fps", type=int, default=30, help="保存视频的帧率")
    parser.add_argument("--save_video", default=None, help="保存 mp4 视频的路径，不填则不保存")
    parser.add_argument("--no_viewer", action="store_true",
                        help="不弹出交互式 MuJoCo 查看器窗口（无显示环境下配合 --save_video 使用）")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--realtime", action="store_true",
                        help="交互查看器按 fps 节流播放（默认尽快播放，配合 viewer 观察请开启）")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl" if args.no_viewer else os.environ.get("MUJOCO_GL", "glfw"))

    print(f"[加载模型] {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    q_start = load_start_qpos(args.start_pose_file)
    print(f"[起始位姿] 来自 {args.start_pose_file}\n  joint_angles_rad = {q_start.tolist()}")
    set_arm_qpos(model, data, q_start)
    set_gripper_qpos(model, data, 0.0)
    mujoco.mj_forward(model, data)

    T_base_world = get_body_T_world(model, data, BASE_BODY_NAME)
    T_tcp_flange = load_tcp_flange(args.tcp_flange_file)
    print(f"[TCP-法兰标定] 来自 {args.tcp_flange_file}\n{T_tcp_flange}")

    grasp_pose_files = {
        "square_nut": args.grasp_pose_file,
        "square_nut_2": args.grasp_pose_file,
        "round_nut": args.round_grasp_pose_file,
    }
    grasp_targets = compute_grasp_targets_from_ground_truth(model, data, here, grasp_pose_files)
    print(f"[抓取目标] 由场景中各螺母的仿真真值位姿直接计算（详见文件顶部说明），"
          f"共 {len(grasp_targets)} 个目标: {list(grasp_targets.keys())}")

    nut_order = [n.strip() for n in args.nut_order.split(",") if n.strip()]
    for name in nut_order:
        if name not in grasp_targets:
            raise ValueError(f"--nut_order 中的 '{name}' 不是有效的螺母名（{NUT_BODY_NAMES}）")

    # ── 可视化 / 视频保存初始化 ──
    passive_viewer = None
    if not args.no_viewer:
        import mujoco.viewer as mj_viewer
        passive_viewer = mj_viewer.launch_passive(model, data)

    renderer = None
    video_writer = None
    if args.save_video:
        import cv2
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        os.makedirs(os.path.dirname(os.path.abspath(args.save_video)) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.save_video, fourcc, args.fps, (args.width, args.height))
        print(f"[视频] 将保存到 {args.save_video} ({args.width}x{args.height} @ {args.fps}fps)")

    frame_dt = 1.0 / args.fps
    frame_count = {"n": 0}

    def frame_cb(model, data):
        if passive_viewer is not None:
            passive_viewer.sync()
            if args.realtime:
                time.sleep(frame_dt)
        if video_writer is not None:
            import cv2
            renderer.update_scene(data)
            img = renderer.render()
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr)
        frame_count["n"] += 1

    # ── 依次对每个螺母执行 抓取 -> 放置 ──
    q_cur = q_start.copy()
    ratio_cur = 0.0
    for i, name in enumerate(nut_order):
        print(f"\n{'=' * 60}\n[{i + 1}/{len(nut_order)}] 抓取并放置目标: {name}\n{'=' * 60}")
        T_grasp_base = grasp_targets[name]
        waypoints = build_waypoints(T_base_world, T_grasp_base)
        q_cur, ratio_cur = interpolate_and_run(
            model, data, name, q_cur, ratio_cur, waypoints, args.steps_per_segment, frame_cb)
        print(f"  完成 {name}：共渲染 {frame_count['n']} 帧（累计）")

    # ── 全部完成后回到起始姿态 ──
    print(f"\n{'=' * 60}\n[收尾] 全部 {len(nut_order)} 个螺母放置完成，返回起始姿态\n{'=' * 60}")
    q_target = q_start
    for step in range(1, args.steps_per_segment + 1):
        t = step / args.steps_per_segment
        q_interp = q_cur + t * (q_target - q_cur)
        ratio_interp = ratio_cur * (1 - t)
        set_arm_qpos(model, data, q_interp)
        set_gripper_qpos(model, data, ratio_interp)
        mujoco.mj_forward(model, data)
        frame_cb(model, data)

    if video_writer is not None:
        video_writer.release()
        print(f"\n[视频] 已保存 {frame_count['n']} 帧至 {args.save_video}")

    if passive_viewer is not None:
        print("\n[查看器] 动作已播放完毕，窗口将保持打开，按 Ctrl+C 或关闭窗口退出。")
        try:
            while passive_viewer.is_running():
                passive_viewer.sync()
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
        passive_viewer.close()


if __name__ == "__main__":
    main()
