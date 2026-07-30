#!/usr/bin/env python3
"""
09_pick_and_place_nuts_mujoco.py
---------------------------------
Kinova Gen3 + Robotiq 2F-85 + 桌面固定俯视 D435i 相机的"感知(FoundationPose)
位姿估计 + 键盘手动控制机械臂"演示。

本版本按最新需求做了大幅精简：删除了所有与"自动抓取/放置"相关的代码
（IK 求解、抓取候选选择、路点执行、运动学抓取/碰撞开关等），只保留：

1) 【真实物理仿真】gen3_with_gripper_and_nuts.xml 中 3 个螺母已加上
   <freejoint> + 打开碰撞（contype/conaffinity=1），本脚本全程通过
   mujoco.mj_step() 真实积分重力/接触/摩擦力。机械臂/夹爪由位置执行器
   （xml 中已定义的 large_actuator/small_actuator/finger_1/finger_2）驱动。

2) 【螺母摆放 + 物理落稳】程序启动后先让物理仿真运行一段时间
   （--settle_steps）使螺母在重力作用下自然落稳在桌面上。

3) 【桌面固定俯视 D435i 相机】相机 "d435i_top_rgb_camera" 垂直向下看，
   可同时看到 3 个螺母的初始摆放区。

4) 【用真实 FoundationPose 推理得到 6D 位姿】本脚本不直接读取螺母 body
   的仿真真值位姿，而是：
     a) 用俯视相机渲染 RGB/深度/各螺母分割掩码（分割用于 MultiObjectPose-
        Estimator.estimate_all() 的首帧初始化，之后的连续帧改用不依赖
        掩码的 track_all() 做增量跟踪——即"多目标连续估计"）；
     b) 调用 foundationpose_api.MultiObjectPoseEstimator 得到每个螺母在
        俯视相机坐标系下的位姿 T_nut_cam；
     c) 用 T_cam_base（俯视相机相对机械臂基座的固定位姿，由 MuJoCo 正向
        运动学一次性算出，与关节角无关）换算到机械臂基坐标系：
        T_nut_base_est = T_cam_base @ T_nut_cam_est；
     d) 结合预设抓取偏移 T_grasp_nut（nut_grasp_pose.json /
        nut_grasp_pose_round.json）通过坐标变换得到"最终机械臂应到达的
        6D 位姿"（夹爪 TCP 在基坐标系下的位姿）：
        T_grasp_base_est = T_nut_base_est @ T_grasp_nut，
        并进一步换算出末端法兰的 6D 位姿 T_flange_base_est。
   到此为止——本脚本只负责把该 6D 位姿计算出来并打印，不再自动求解 IK、
   不再自动驱动机械臂去抓取。

5) 【相机画面 + 位姿可视化】俯视相机画面全程用 OpenCV 窗口实时显示；
   进入"位姿估计"状态后，每一帧都会在画面上叠加 FoundationPose 估计出的
   3D 包围框 + 坐标轴（MultiObjectPoseEstimator.visualize_all()）。

6) 【连续多目标跟踪】首次按键触发 estimate_all()（需要掩码，较慢但准）
   完成初始化后，后续每一帧都调用 track_all()（不需要掩码，基于上一帧
   位姿做增量跟踪，更快）。

7) 【键盘交互】
     'e' ：开始位姿估计 + 连续跟踪（仅估计+可视化，机械臂不动）
     'r' ：重新初始化位姿估计（跟踪丢失时使用）
     'p' ：（需已按过 'e' 且 3 个螺母都已估计出位姿）通过坐标变换计算并
           打印每个螺母对应的最终机械臂 6D 抓取目标位姿（不驱动机械臂）
     'q' ：退出程序

8) 【新增：键盘手动控制机械臂 + 夹爪】不再有任何自动抓取逻辑，取而代之
   的是随时可用（idle/tracking 任意状态下）的手动关节 + 夹爪控制：
     '1'-'7'      ：选中要控制的关节编号（joint_1 ~ joint_7）
     '['          ：选中关节角度减小一个步长
     ']'          ：选中关节角度增大一个步长
     'c'          ：夹爪闭合一个步长
     'o'          ：夹爪张开一个步长
   每次按键会直接把对应关节/夹爪执行器的目标 ctrl 值增/减一个固定步长，
   真实物理仿真持续运行（mj_step 每帧都会调用），因此手动调整的关节会
   平滑地运动到新的目标角度，而不是瞬间跳变。

── 抓取偏移 / TCP-法兰标定 ─────────────────────────────────────────────
T_grasp_nut（夹爪 TCP 相对螺母局部坐标系的抓取偏移）与 T_tcp_flange（TCP
相对末端法兰的固定偏移）均沿用之前已验证过的定义文件（nut_grasp_pose.json /
nut_grasp_pose_round.json / tcp_flange.json），仅用于坐标变换计算+打印，
未做改动。

用法示例
--------
  # 交互式（需要图形界面）：先看相机画面，按 e 估计位姿，按 p 打印 6D 目标位姿，
  # 同时可以随时用数字键+[]/c/o 手动控制机械臂和夹爪
  python 09_pick_and_place_nuts_mujoco.py --viewer

  # 无图形界面/自动化测试：自动模拟按键 'e' 后持续跟踪+打印
  python 09_pick_and_place_nuts_mujoco.py --no_gui --auto --save_video output_3nuts/pose_estimation_vision.mp4
"""

import argparse
import json
import math
import os
import threading
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from foundationpose_api import MultiObjectPoseEstimator

# ─────────────────────────────────────────────────────────────────────────────
ARM_JOINT_NAMES = ["joint_1", "joint_2", "joint_3",
                   "joint_4", "joint_5", "joint_6", "joint_7"]
# Robotiq 2F-85 由 finger_1 / finger_2 两个位置执行器直接驱动
# （right_outer_knuckle_joint 通过 xml 中的 <tendon> 与另外 4 个从动关节
# 耦合，真实物理仿真下无需像旧版脚本那样手动改写全部 6 个关节 qpos）。
GRIPPER_ACTUATOR_NAMES = ["finger_1", "finger_2"]
GRIPPER_ACTUATOR_MAX = 0.8  # 对应 xml 中 finger_1/finger_2 的 ctrlrange 上限

GRIP_SITE = "grip_site"
BASE_BODY_NAME = "base_link"
TOP_CAMERA_NAME = "d435i_top_rgb_camera"

NUT_BODY_NAMES = ["square_nut", "square_nut_2", "round_nut"]
DEFAULT_GRASP_POSE_FILES = {
    "square_nut": "nut_grasp_pose.json",
    "square_nut_2": "nut_grasp_pose.json",
    "round_nut": "nut_grasp_pose_round.json",
}

# ── 手动关节 / 夹爪控制的步长（每按一次键改变的量）───────────────────────
MANUAL_JOINT_STEP_DEG = 2.0      # 每次按 '[' / ']' 调整关节角度的步长（度）
MANUAL_GRIPPER_STEP = 0.05       # 每次按 'c' / 'o' 调整夹爪开合比例的步长（0~1）

# ── 位姿估计鲁棒性过滤（解决"没有目标的位置也出现目标"的异常检测问题）───
# 螺母分割掩码有效像素数低于此阈值时，视为"当前帧看不清/被遮挡"，直接跳过
# 本帧对该目标的 estimate_all() 调用（避免用一小撮噪声像素去初始化出一个
# 完全错误的位姿——mask 太小/太碎时 FoundationPose 的 PnP/优化很容易收敛到
# 错误极值，导致包围框出现在画面中根本没有物体的位置）。
MIN_MASK_PIXELS = 300
# 俯视相机固定安装高度（xml 中 top_d435i_mount pos z=1.6）减去桌面高度
# （table 上表面 z=0.78）得到的名义工作距离，用于合理性校验：一个真正贴在
# 桌面上的螺母，其在相机坐标系下的 Z（深度，米）应该落在下方区间内；明显
# 超出此区间的估计结果视为跟踪/估计出现异常（例如收敛到了错误的局部最优、
# 或目标被遮挡后 track() 仍在对着错误的背景像素做增量跟踪），应被丢弃而不是
# 继续显示/参与后续的坐标变换计算。
CAM_HEIGHT_ABOVE_TABLE_M = 1.6 - 0.78
DEPTH_SANITY_RANGE_M = (CAM_HEIGHT_ABOVE_TABLE_M - 0.25, CAM_HEIGHT_ABOVE_TABLE_M + 0.25)

# MuJoCo 相机沿 -Z 看（OpenGL 约定）→ OpenCV 相机 +Z 朝前，相差绕 X 轴 180°
T_FLIP = np.eye(4)
T_FLIP[:3, :3] = np.diag([1.0, -1.0, -1.0])


# ─────────────────────────────────────────────────────────────────────────────
# 基础工具函数
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


def get_site_T_world(model, data, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"找不到 site: {site_name}")
    return make_T(data.site_xpos[site_id], data.site_xmat[site_id])


def get_cam_T_world_cv(model, data, camera_name):
    """相机在世界坐标系下的位姿（OpenCV 约定：+Z 朝前），与
    08_compute_nut_pose_in_base.py 中同名函数完全一致的换算方式。"""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id == -1:
        raise ValueError(f"找不到相机: {camera_name}")
    T_cam_world_mj = make_T(data.cam_xpos[cam_id], data.cam_xmat[cam_id])
    return T_cam_world_mj @ T_FLIP


def get_camera_intrinsics(model, cam_name, width, height):
    """与 05_collect_pose_estimation_data_mujoco.py 中同名函数完全一致：
    由 MuJoCo 相机的 fovy（垂直视场角）+ 目标分辨率反推针孔相机内参
    （方形像素假设，fx == fy，无畸变）。"""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1:
        raise ValueError(f"找不到相机: {cam_name}")
    fovy_rad = model.cam_fovy[cam_id] * math.pi / 180.0
    fy = (height / 2.0) / math.tan(fovy_rad / 2.0)
    fx = fy
    K = np.array([[fx, 0.0, width / 2.0],
                  [0.0, fy, height / 2.0],
                  [0.0, 0.0, 1.0]])
    return K


def get_all_nut_geom_ids(model, nut_names):
    """返回 {螺母名: geom id 集合}，与 05_collect_pose_estimation_data_mujoco.py
    中同名函数完全一致，用于从分割渲染结果中抠出每个螺母各自的掩码。"""
    geom_ids = {}
    for name in nut_names:
        nut_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if nut_id == -1:
            raise ValueError(f"找不到 body: {name}")
        start = model.body_geomadr[nut_id]
        count = model.body_geomnum[nut_id]
        geom_ids[name] = set(range(start, start + count))
    return geom_ids


def render_rgb_depth_masks(renderer, data, camera_name, nut_geom_ids, depth_max_m=5.0):
    """在同一个 update_scene() 之后依次渲染 RGB（RGB 通道顺序） / 深度（米）/
    各螺母分割掩码，与 05_collect_pose_estimation_data_mujoco.py 中
    render_rgbd_mask() 的实现方式完全一致（同一个 Renderer 对象在
    rgb/depth/segmentation 三种模式间切换复用，避免重复创建渲染上下文）。"""
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    depth_m = renderer.render().copy()
    renderer.disable_depth_rendering()
    depth_m[(depth_m < 0.001) | (depth_m > depth_max_m)] = 0.0

    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    obj_id = seg[:, :, 0]
    obj_type = seg[:, :, 1]
    is_geom = obj_type == mujoco.mjtObj.mjOBJ_GEOM

    masks_u8 = {}
    for name, geom_ids in nut_geom_ids.items():
        nut_mask = is_geom & np.isin(obj_id, list(geom_ids))
        masks_u8[name] = (nut_mask.astype(np.uint8)) * 255

    return rgb, depth_m, masks_u8


def get_joint_qpos_adr(model, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_qposadr[jid]


def get_joint_dof_adr(model, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_dofadr[jid]


def set_arm_qpos(model, data, q7):
    for name, q in zip(ARM_JOINT_NAMES, q7):
        data.qpos[get_joint_qpos_adr(model, name)] = q


def get_arm_qpos(model, data):
    return np.array([data.qpos[get_joint_qpos_adr(model, n)] for n in ARM_JOINT_NAMES])


def set_arm_ctrl(model, data, q7):
    for name, q in zip(ARM_JOINT_NAMES, q7):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        data.ctrl[aid] = q


def set_gripper_ctrl_ratio(model, data, ratio):
    """ratio=0 -> 完全张开，ratio=1 -> 完全闭合（ctrl = ratio * 0.8）。
    真实物理仿真下只需驱动 finger_1/finger_2 两个执行器，其余 4 个从动
    关节由 xml 中的 <tendon> 被动耦合传动，无需手动写 qpos。"""
    for name in GRIPPER_ACTUATOR_NAMES:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        data.ctrl[aid] = ratio * GRIPPER_ACTUATOR_MAX


def rotation_log(R):
    return Rotation.from_matrix(R).as_rotvec()


def load_start_qpos(start_pose_file):
    with open(start_pose_file, encoding="utf-8") as f:
        d = json.load(f)
    return np.array(d["joint_angles_rad"], dtype=float)


def load_tcp_flange(tcp_flange_file):
    with open(tcp_flange_file, encoding="utf-8") as f:
        d = json.load(f)
    return np.array(d["T_tcp_flange"])


def load_grasp_pose(grasp_pose_file):
    """返回该文件里的抓取候选位姿列表（4x4 矩阵的 list）。若文件里有
    "T_grasp_nut_candidates"（新格式，见 nut_grasp_pose.json 顶部注释），
    优先使用其 "candidates" 数组；否则退回旧格式，只用单个 "T_grasp_nut"
    包装成长度为 1 的列表，保持向后兼容。"""
    with open(grasp_pose_file, encoding="utf-8") as f:
        grasp = json.load(f)
    if "T_grasp_nut_candidates" in grasp:
        return [np.array(c) for c in grasp["T_grasp_nut_candidates"]["candidates"]]
    return [np.array(grasp["T_grasp_nut"])]


def filter_valid_masks(masks, min_pixels=MIN_MASK_PIXELS):
    """丢弃像素数过少（噪声/严重遮挡）的掩码，避免用碎片掩码去初始化位姿
    估计而产生完全错误的结果。返回值可直接传给 estimate_all()。"""
    valid = {}
    for name, mask in masks.items():
        if mask is not None and int(np.count_nonzero(mask)) >= min_pixels:
            valid[name] = mask
    return valid


def fix_flat_object_updown_ambiguity(T_nut_base):
    """修正"薄片状对称螺母"常见的 180° 上下翻转歧义。

    背景（"机械臂抓取位置根本不是 nut 真实位置"问题的根因）：
    通过对比 FoundationPose 视觉估计位姿与仿真真值位姿（同一 base_link
    坐标系下）发现：位置（平移）估计误差始终 <1.5mm，完全正确；但
    square_nut / square_nut_2 的旋转误差高达 ~180°（round_nut 因绕通孔轴
    旋转对称而没有这个问题，误差仅 0.8°）。

    原因：square_nut/square_nut_2 是薄片状物体（沿局部 Z 轴很薄），从俯视
    相机（几乎垂直向下）看过去，"正面朝上"和"整体绕水平轴翻转 180°（即
    正面朝下、背面朝上）"两种姿态在轮廓/深度图上几乎完全一样 —— 这是薄片
    对称物体本身固有的姿态歧义，不是坐标变换或坐标系定义的 bug（平移误差
    <1.5mm 已经证明整条坐标变换链路 T_cam_base/T_nut_cam/矩阵乘法 都是对的）。

    该歧义体现为：估计出的螺母局部 Z 轴（螺母厚度方向法线，nut_grasp_pose.json
    中 T_grasp_nut 的抓取偏移正是按"局部 +Z 朝上"这个约定标定的）在世界坐标系
    下指向下方（穿入桌面）而不是上方——这在物理上不可能（螺母平放在桌面上，
    法线必然朝上），是一个可以事后按几何规则直接修正的、明确的错误分支，
    而不需要修改感知模型本身。

    修正方法：若估计位姿的局部 Z 轴（旋转矩阵第 3 列）在世界系下的 Z 分量
    为负（朝下），说明命中了"翻转过来"的这一支解，绕该位姿的局部 X 轴
    旋转 180°，把 Z 轴重新翻回朝上，即可恢复与标定文件一致的"正面朝上"
    约定，抓取偏移 T_grasp_nut 就能正确复合到把手上真正所在的位置。
    """
    R = T_nut_base[:3, :3]
    local_z_in_world = R[:, 2]
    if local_z_in_world[2] < 0.0:
        R_flip_x = Rotation.from_rotvec(np.array([np.pi, 0.0, 0.0])).as_matrix()
        T_fixed = T_nut_base.copy()
        T_fixed[:3, :3] = R @ R_flip_x
        return T_fixed
    return T_nut_base


def pose_is_plausible(T_nut_cam, depth_range_m=DEPTH_SANITY_RANGE_M):
    """粗略合理性校验：一个真正贴在桌面上的螺母，其在相机坐标系下的 Z（深度）
    应落在“俯视相机安装高度 ± 容差”范围内。用于过滤 FoundationPose 估计/
    跟踪偶尔收敛到错误位置（例如跟丢后对着背景像素继续跟踪）产生的“幽灵”
    位姿——这类异常位姿的深度往往会明显偏离桌面所在的深度范围。"""
    if T_nut_cam is None:
        return False
    z_cam = float(T_nut_cam[2, 3])
    return depth_range_m[0] <= z_cam <= depth_range_m[1]


# ─────────────────────────────────────────────────────────────────────────────
# 主程序：交互式状态机
# ─────────────────────────────────────────────────────────────────────────────

class Demo:
    def __init__(self, args):
        self.args = args
        here = os.path.dirname(os.path.abspath(args.xml)) or "."
        self.here = here

        print(f"[加载模型] {args.xml}")
        self.model = mujoco.MjModel.from_xml_path(args.xml)

        # ── 关闭离屏渲染的多重采样抗锯齿（MSAA，xml 默认 offsamples=4）──
        # 根因排查记录：按 E 之后 square_nut 估计正确，但 square_nut_2 /
        # round_nut 的位姿明显错误（3D 框和坐标轴出现在错误位置）。逐层排查
        # （对比仿真真值 vs FoundationPose 实际输出 → 怀疑多目标共享
        # scorer/refiner/glctx 导致串扰 → 改用完全独立、互不共享的估计器
        # 实例复测，结果仍然错误，且数值与之前几乎一致 → 怀疑是物体距相机
        # 主光轴过远导致的裁剪/投影问题 → 把物体裁剪、重新居中到画面正中央
        # 复测，结果依然一样错误 → 直接在 FoundationPose 内部
        # `guess_translation()`（据"分割掩码的像素外接矩形"估算的平移种子值）
        # 打点，发现这个"种子值"本身就已经是错的 → 直接检查分割掩码本身的
        # 外接矩形，发现 square_nut_2 的分割掩码里混入了一些离该螺母真实
        # 轮廓非常远的"杂散像素"（例如某个子 geom 的 x 像素范围本该只有
        # 30 多像素宽，实际却横跨了 300 多像素）。
        # 根本原因：MuJoCo 离屏渲染默认开启了 4x MSAA（多重采样抗锯齿，
        # model.vis.quality.offsamples 默认为 4）。分割渲染（segmentation
        # rendering）依赖把"每个物体的整数 ID"编码进特定的颜色通道里，靠
        # 精确解码颜色数值还原 ID；但 MSAA 会在物体边缘对相邻像素的颜色做
        # 插值混合以实现抗锯齿效果，边缘处混合出来的颜色不再对应任何一个
        # 真实存在的 ID，解码出来就是一个随机的、可能落在画面任意位置的
        # "垃圾"整数值——这正是 square_nut_2 的掩码里混入大量离谱杂散像素、
        # 进而导致 guess_translation() 算出的初始平移种子严重偏离真实位置的
        # 根本原因（已通过设置 offsamples=0 后重新渲染验证：所有杂散像素
        # 消失，3 个螺母的分割掩码外接矩形全部恢复正常紧凑范围，且
        # FoundationPose 位姿估计误差从原先的 0.3+ 米降至 <2 mm）。
        # 因此这里显式关闭 MSAA，确保分割渲染出的整数 ID 精确可靠。这不会
        # 影响 RGB/深度渲染的视觉质量（本场景对 RGB 抗锯齿要求不高），只是
        # 让分割通道的解码结果保持像素级精确。
        self.model.vis.quality.offsamples = 0

        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.q_start = load_start_qpos(args.start_pose_file)
        set_arm_qpos(self.model, self.data, self.q_start)
        set_arm_ctrl(self.model, self.data, self.q_start)
        set_gripper_ctrl_ratio(self.model, self.data, 0.0)
        mujoco.mj_forward(self.model, self.data)

        self.T_tcp_flange = load_tcp_flange(args.tcp_flange_file)
        self.grasp_pose_files = {
            "square_nut": args.grasp_pose_file,
            "square_nut_2": args.grasp_pose_file,
            "round_nut": args.round_grasp_pose_file,
        }
        self.mesh_files = {
            "square_nut": args.mesh,
            "square_nut_2": args.mesh,
            "round_nut": args.round_mesh,
        }
        self.nut_order = [n.strip() for n in args.nut_order.split(",") if n.strip()]

        self.nut_geom_ids = get_all_nut_geom_ids(self.model, NUT_BODY_NAMES)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.K = get_camera_intrinsics(self.model, TOP_CAMERA_NAME, 640, 480)
        print(f"[俯视相机内参] fovy={self.model.cam_fovy[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, TOP_CAMERA_NAME)]:.1f}°\n{self.K}")

        # T_cam_base：俯视相机相对机械臂基座的固定位姿（与关节角无关，因为
        # 相机和 base_link 都是固定在世界坐标系中的静态 body）。
        T_base_world = get_body_T_world(self.model, self.data, BASE_BODY_NAME)
        T_cam_world_cv = get_cam_T_world_cv(self.model, self.data, TOP_CAMERA_NAME)
        self.T_cam_base = np.linalg.inv(T_base_world) @ T_cam_world_cv
        self.T_base_world = T_base_world

        self.estimator = None  # 首次按 'e' 时才加载权重（懒加载）
        self.latest_poses = {}  # name -> T_nut_cam (4x4)，最近一次估计/跟踪结果
        self.state = "idle"     # idle -> tracking（不再有 executing/done：手动控制随时可用）
        self._just_entered_tracking = False

        # ── 异步推理相关状态（解决"按 E 之后窗口卡死/系统提示无响应"问题）──
        # 首次加载 FoundationPose 权重 + 首帧 estimate_all() register 通常需要
        # 数秒到一分钟不等，如果在主线程里同步阻塞执行，期间既不刷新画面也
        # 不处理 cv2 窗口消息循环，操作系统会认为窗口"未响应"。这里把"权重
        # 加载 + estimate_all()"这一耗时操作放到后台线程执行，主线程继续以
        # 正常帧率渲染相机画面 + 调用 cv2.waitKey()（保持窗口消息循环畅通），
        # 并在画面上叠加"正在估计位姿，请稍候…"提示，每帧轮询后台线程是否
        # 完成。注意：逐帧 track_all() 本身已经是为实时调用设计的（较快），
        # 因此不需要异步化，仍在主线程同步调用。
        self._infer_thread = None
        self._infer_busy = False
        self._infer_result = None
        self._infer_error = None

        self.video_writer = None
        if args.save_video:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_video)) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(args.save_video, fourcc, args.fps, (640, 480))
            print(f"[视频] 将保存到 {args.save_video} (640x480 @ {args.fps}fps)")

        self.viewer = None
        if args.viewer:
            import mujoco.viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.model, self.data)

        self.window_name = "Top D435i Camera - FoundationPose 6D Pose"
        self.frame_idx = 0
        self._quit = False

        # ── 手动控制状态（替代原先的自动抓取/放置逻辑）───────────────────
        # active_joint_idx：当前被 '[' / ']' 键控制的关节，取值 0~6，
        # 对应 ARM_JOINT_NAMES[0..6]（joint_1 ~ joint_7）。
        # q_manual：手动控制下机械臂 7 个关节的目标角度（ctrl 目标值），
        # 初始等于起始姿态 q_start；每次按 '[' / ']' 就地增减对应分量后
        # 立即调用 set_arm_ctrl() 下发，真实物理会平滑地驱动过去。
        # gripper_ratio：手动控制下夹爪开合比例（0=张开, 1=闭合）。
        self.active_joint_idx = 0
        self.q_manual = self.q_start.copy()
        self.gripper_ratio = 0.0

    # ── 后台线程：加载权重（如需要）+ 执行一次 estimate_all() ──────────────
    def _start_estimate_async(self, rgb, masks, depth_m):
        self._infer_busy = True
        self._infer_result = None
        self._infer_error = None

        def worker(rgb=rgb, masks=masks, depth_m=depth_m):
            try:
                if self.estimator is None:
                    print("[FoundationPose] 首次使用，正在加载权重（较慢，请稍候）…")
                    self.estimator = MultiObjectPoseEstimator(
                        objects=self.mesh_files, K=self.K,
                        weights_dir=self.args.weights_dir,
                        est_refine_iter=self.args.est_refine_iter,
                        track_refine_iter=self.args.track_refine_iter,
                        debug=0, debug_dir="output_vision",
                    )
                print("[FoundationPose] estimate_all()：首帧多目标初始化 …")
                valid_masks = filter_valid_masks(masks)
                skipped = set(masks.keys()) - set(valid_masks.keys())
                if skipped:
                    print(f"  [跳过] 掩码像素数不足（可能被遮挡/暂不可见），本帧不估计: {sorted(skipped)}")
                self._infer_result = self.estimator.estimate_all(rgb, valid_masks, depth=depth_m)
            except Exception as e:  # noqa: BLE001 - 后台线程需要把异常带回主线程打印
                print(f"[FoundationPose] estimate_all() 失败: {e}")
                self._infer_error = e
                self._infer_result = {}
            finally:
                self._infer_busy = False

        self._infer_thread = threading.Thread(target=worker, daemon=True)
        self._infer_thread.start()

    def _apply_poses(self, poses):
        """对一批新估计/跟踪出的位姿做合理性过滤后合并进 self.latest_poses。"""
        for name, T_nut_cam in list(poses.items()):
            if not pose_is_plausible(T_nut_cam):
                print(f"  [异常位姿丢弃] '{name}' 深度={T_nut_cam[2, 3]:.3f}m 超出合理范围 "
                      f"{DEPTH_SANITY_RANGE_M}，已重置该目标跟踪状态。")
                del poses[name]
                self.latest_poses.pop(name, None)
                if self.estimator is not None:
                    self.estimator.reset(name)
        self.latest_poses.update(poses)

    # ── 渲染 + 状态机推进 + 显示/录制（每个"显示帧"调用一次）──────────────
    def render_and_display(self, status_lines=None):
        rgb, depth_m, masks = render_rgb_depth_masks(
            self.renderer, self.data, TOP_CAMERA_NAME, self.nut_geom_ids)

        waiting_text = None
        if self.state == "tracking":
            if self._infer_busy:
                # 后台线程仍在跑（权重加载 / estimate_all），本帧只显示原始
                # 画面 + 提示文字，绝不在主线程里做任何阻塞调用，确保
                # cv2.waitKey() 每帧都能被调用到，窗口消息循环不会卡死。
                waiting_text = "Estimating pose... please wait (first time may take up to ~1 min to load weights)"
            elif self._just_entered_tracking:
                if self._infer_thread is None:
                    # 尚未启动后台估计线程：启动它，本帧先显示提示文字。
                    self._start_estimate_async(rgb, masks, depth_m)
                    waiting_text = "Starting pose estimation..."
                else:
                    # 后台线程已完成（_infer_busy 刚变为 False），取出结果。
                    self._apply_poses(self._infer_result or {})
                    self._infer_thread = None
                    self._just_entered_tracking = False
            else:
                # 正常连续跟踪帧：track_all() 本身按实时调用设计，速度较快，
                # 保持同步调用（这是需求里"连续多目标跟踪"的核心）。
                poses = self.estimator.track_all(rgb, depth=depth_m)
                self._apply_poses(poses)

            if self.estimator is not None and self.latest_poses:
                vis_bgr = self.estimator.visualize_all(rgb, self.latest_poses)
            else:
                vis_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if waiting_text:
                cv2.putText(vis_bgr, waiting_text, (8, 460), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis_bgr, waiting_text, (8, 460), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            vis_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        lines = list(status_lines or [])
        active_joint_name = ARM_JOINT_NAMES[self.active_joint_idx]
        lines.append(f"state={self.state}  tracked={list(self.latest_poses.keys())}")
        lines.append(f"[manual] active_joint={active_joint_name} ({self.active_joint_idx + 1}/7) "
                     f"angle={math.degrees(self.q_manual[self.active_joint_idx]):.1f}deg  "
                     f"gripper={self.gripper_ratio:.2f}")
        lines.append("[1-7] select joint  [ [/] ] +-joint  [c] close gripper  [o] open gripper")
        if self.state == "idle":
            lines.append("[e] start pose estimation   [q] quit")
        elif self.state == "tracking":
            ready = all(n in self.latest_poses for n in self.nut_order)
            lines.append(f"[p] {'print 6D grasp pose (ready)' if ready else 'print 6D grasp pose (waiting for all nuts)'}   [r] reinit   [q] quit")
        for i, line in enumerate(lines):
            cv2.putText(vis_bgr, line, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis_bgr, line, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if not self.args.no_gui:
            cv2.imshow(self.window_name, vis_bgr)
        if self.video_writer is not None:
            self.video_writer.write(vis_bgr)

        key = self._poll_key()
        self.frame_idx += 1
        return key

    def _poll_key(self):
        if self.args.no_gui:
            return -1
        k = cv2.waitKey(1) & 0xFF
        return k if k != 255 else -1

    # ── 手动关节 / 夹爪控制（随时可用，不受 idle/tracking 状态影响）───────
    def _apply_manual_joint_delta(self, delta_rad):
        idx = self.active_joint_idx
        self.q_manual[idx] += delta_rad
        set_arm_ctrl(self.model, self.data, self.q_manual)
        print(f"    [手动控制] {ARM_JOINT_NAMES[idx]} -> {math.degrees(self.q_manual[idx]):.1f}deg")

    def _apply_manual_gripper_delta(self, delta_ratio):
        self.gripper_ratio = float(np.clip(self.gripper_ratio + delta_ratio, 0.0, 1.0))
        set_gripper_ctrl_ratio(self.model, self.data, self.gripper_ratio)
        print(f"    [手动控制] 夹爪比例 -> {self.gripper_ratio:.2f} "
              f"(0=张开, 1=闭合)")

    def handle_key(self, key):
        if key in (ord('q'), ord('Q')):
            self._quit = True
        elif key in (ord('e'), ord('E')) and self.state == "idle":
            print("[状态] idle -> tracking（开始位姿估计 + 连续跟踪）")
            self.state = "tracking"
            self._just_entered_tracking = True
        elif key in (ord('r'), ord('R')) and self.state == "tracking":
            if self._infer_busy:
                print("[提示] 上一次位姿估计仍在后台运行中，请稍候再重新初始化。")
            else:
                print("[状态] 重新初始化位姿估计")
                self.latest_poses.clear()
                self._infer_thread = None
                self._just_entered_tracking = True
        elif key in (ord('p'), ord('P')) and self.state == "tracking":
            if self._infer_busy:
                print("[提示] 位姿估计仍在后台运行中，请稍候再打印 6D 位姿。")
            elif all(n in self.latest_poses for n in self.nut_order):
                self.compute_and_print_final_poses()
            else:
                print("[提示] 还有螺母未成功估计出位姿，暂不能计算最终 6D 位姿。")
        elif key in tuple(ord(str(d)) for d in range(1, 8)):
            self.active_joint_idx = int(chr(key)) - 1
            print(f"    [手动控制] 已选中关节 {ARM_JOINT_NAMES[self.active_joint_idx]}")
        elif key == ord('['):
            self._apply_manual_joint_delta(-math.radians(MANUAL_JOINT_STEP_DEG))
        elif key == ord(']'):
            self._apply_manual_joint_delta(math.radians(MANUAL_JOINT_STEP_DEG))
        elif key in (ord('c'), ord('C')):
            self._apply_manual_gripper_delta(MANUAL_GRIPPER_STEP)
        elif key in (ord('o'), ord('O')):
            self._apply_manual_gripper_delta(-MANUAL_GRIPPER_STEP)

    # ── 让物理仿真先运行一段时间，使螺母在重力下落稳到桌面 ────────────────
    def settle(self):
        print(f"[物理落稳] 运行 {self.args.settle_steps} 个仿真步，让螺母在重力下落稳到桌面 …")
        display_every = max(1, round((1.0 / self.args.fps) / self.model.opt.timestep))
        for step in range(self.args.settle_steps):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()
            if step % display_every == 0:
                key = self.render_and_display(status_lines=["settling physics ..."])
                self.handle_key(key)
                if self._quit:
                    return

    # ── 根据当前 latest_poses，通过坐标变换计算并打印每个螺母对应的最终
    # 机械臂 6D 抓取目标位姿（不求解 IK、不驱动机械臂，只输出数值）───────
    def compute_and_print_final_poses(self):
        """对 nut_order 中每个螺母：
          1) T_nut_base_est = T_cam_base @ T_nut_cam（俯视相机估计位姿换算到
             机械臂基坐标系下）；
          2) 修正薄片对称螺母的 180° 上下翻转姿态歧义；
          3) T_grasp_base_est = T_nut_base_est @ T_grasp_nut（复合抓取偏移，
             得到夹爪 TCP 在基坐标系下的最终 6D 位姿）；
          4) T_flange_base_est = T_grasp_base_est @ inv(T_tcp_flange)（进一步
             换算出末端法兰的 6D 位姿）。
        每个螺母只取 nut_grasp_pose*.json 中的第一个抓取候选（不做 IK 可行性
        筛选——本脚本已不再自动求解 IK/自动抓取，仅负责把坐标变换结果算出来
        并打印，如何使用该结果（比如挑选/求解 IK）交由后续手动或其他脚本
        处理）。同时打印与仿真真值的 pos_err/rot_err 作为精度参考。"""
        print(f"\n{'=' * 60}\n[6D 位姿输出] 通过坐标变换计算最终机械臂抓取目标位姿: {self.nut_order}\n{'=' * 60}")
        for name in self.nut_order:
            T_nut_cam = self.latest_poses[name]
            T_nut_base_est = self.T_cam_base @ T_nut_cam
            T_nut_base_est = fix_flat_object_updown_ambiguity(T_nut_base_est)

            grasp_file = self.grasp_pose_files[name]
            if not os.path.isabs(grasp_file):
                grasp_file = os.path.join(self.here, grasp_file)
            T_grasp_nut_candidates = load_grasp_pose(grasp_file)
            T_grasp_nut = T_grasp_nut_candidates[0]
            T_grasp_base_est = T_nut_base_est @ T_grasp_nut
            T_flange_base_est = T_grasp_base_est @ np.linalg.inv(self.T_tcp_flange)

            T_nut_world_gt = get_body_T_world(self.model, self.data, name)
            T_nut_base_gt = np.linalg.inv(self.T_base_world) @ T_nut_world_gt
            pos_err = np.linalg.norm(T_nut_base_est[:3, 3] - T_nut_base_gt[:3, 3])
            rot_err_deg = np.degrees(np.linalg.norm(
                rotation_log(T_nut_base_est[:3, :3] @ T_nut_base_gt[:3, :3].T)))

            grasp_xyz = T_grasp_base_est[:3, 3]
            grasp_rpy_deg = Rotation.from_matrix(T_grasp_base_est[:3, :3]).as_euler("xyz", degrees=True)
            grasp_quat_xyzw = Rotation.from_matrix(T_grasp_base_est[:3, :3]).as_quat()
            flange_xyz = T_flange_base_est[:3, 3]
            flange_rpy_deg = Rotation.from_matrix(T_flange_base_est[:3, :3]).as_euler("xyz", degrees=True)

            print(f"\n[{name}]")
            print(f"  [坐标校验] pos_err={pos_err * 1000:.1f}mm  rot_err={rot_err_deg:.1f}deg (对比仿真真值)")
            print(f"  夹爪TCP 6D位姿 (base系): xyz(m)={grasp_xyz}  rpy(deg)={grasp_rpy_deg}  "
                  f"quat(xyzw)={grasp_quat_xyzw}")
            print(f"  末端法兰 6D位姿 (base系): xyz(m)={flange_xyz}  rpy(deg)={flange_rpy_deg}")
            print(f"  T_grasp_base=\n{T_grasp_base_est}")
            print(f"  T_flange_base=\n{T_flange_base_est}")

    # ── 主循环：idle / tracking 状态下等待按键；手动控制随时生效 ──────────
    # 与此前"只在 settle()/抓取路点执行期间才 mj_step()"不同，删除自动抓取
    # 逻辑后，机械臂运动完全由手动按键触发，因此这里的主循环需要在每个
    # 显示帧之间持续推进物理仿真（否则手动调整 ctrl 目标后，手臂不会真的
    # 动起来），使按键控制的效果能被实时看到。
    def run(self):
        self.settle()
        display_every = max(1, round((1.0 / self.args.fps) / self.model.opt.timestep))
        auto_frame_counter = 0
        while not self._quit:
            for _ in range(display_every):
                mujoco.mj_step(self.model, self.data)
                if self.viewer is not None:
                    self.viewer.sync()

            key = self.render_and_display()
            self.handle_key(key)

            if self.args.auto:
                auto_frame_counter += 1
                if self.state == "idle" and auto_frame_counter >= self.args.auto_estimate_after:
                    self.handle_key(ord('e'))
                elif self.state == "tracking" and auto_frame_counter >= (
                        self.args.auto_estimate_after + self.args.auto_print_after):
                    if all(n in self.latest_poses for n in self.nut_order):
                        self.compute_and_print_final_poses()
                        self._quit = True  # 自动模式下打印完 6D 位姿即退出

            if not self.args.no_gui:
                time.sleep(0.0)  # 让出时间片，cv2.waitKey 已在 render_and_display 中处理节流

        self.cleanup()

    def cleanup(self):
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"\n[视频] 已保存至 {self.args.save_video}")
        if self.viewer is not None:
            self.viewer.close()
        if not self.args.no_gui:
            cv2.destroyAllWindows()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Gen3 + Robotiq 2F-85 + 俯视 D435i 相机：FoundationPose 位姿估计 + 键盘手动控制机械臂/夹爪",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xml", default=os.path.join(here, "gen3_with_gripper_and_nuts.xml"))
    parser.add_argument("--start_pose_file",
                        default=os.path.join(here, "pose_estimation_data_3nuts", "poses", "frame_000000.json"))
    parser.add_argument("--mesh", default=os.path.join(here, "nut_mesh", "textured_simple.obj"),
                        help="方形螺母（square_nut/square_nut_2）共用网格")
    parser.add_argument("--round_mesh", default=os.path.join(here, "nut_mesh", "round_nut_textured_simple.obj"),
                        help="圆形螺母（round_nut）网格")
    parser.add_argument("--grasp_pose_file", default=os.path.join(here, "nut_grasp_pose.json"))
    parser.add_argument("--round_grasp_pose_file", default=os.path.join(here, "nut_grasp_pose_round.json"))
    parser.add_argument("--tcp_flange_file", default=os.path.join(here, "tcp_flange.json"))
    parser.add_argument("--weights_dir", default=os.path.join(here, "weights"))
    parser.add_argument("--nut_order", default="square_nut,square_nut_2,round_nut")

    parser.add_argument("--settle_steps", type=int, default=800,
                        help="程序启动后先运行多少个物理步让螺母落稳到桌面")
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)

    parser.add_argument("--fps", type=int, default=30, help="显示/录制视频的帧率")
    parser.add_argument("--save_video", default=None, help="保存俯视相机画面(含位姿可视化)mp4路径")
    parser.add_argument("--viewer", action="store_true", help="额外弹出第三人称 MuJoCo 交互查看器窗口")
    parser.add_argument("--no_gui", action="store_true",
                        help="不弹出 OpenCV 相机画面窗口（无显示环境下使用，配合 --auto）")
    parser.add_argument("--auto", action="store_true",
                        help="自动模拟按键（先'e'后打印 6D 位姿），用于无人值守/无显示环境测试位姿估计链路")
    parser.add_argument("--auto_estimate_after", type=int, default=30,
                        help="--auto 模式下，等待多少显示帧后自动按 'e'")
    parser.add_argument("--auto_print_after", type=int, default=60,
                        help="--auto 模式下，进入 tracking 状态后再等待多少显示帧自动打印 6D 位姿")
    args = parser.parse_args()

    if not args.no_gui:
        os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "glfw")

    demo = Demo(args)
    demo.run()


if __name__ == "__main__":
    main()
