#!/usr/bin/env python3
"""
sim_common.py
-------------
【非算法基础设施代码】——从 10_pick_and_place_nuts_mujoco.py 中拆分出来的
"与感知/抓取规划控制算法无关"的公共基础设施：MuJoCo 模型访问、坐标系/
位姿工具函数、机械臂关节/夹爪底层控制原语、标定文件加载、3D 可视化绘制
辅助函数、场景相关常量等。

设计目的
--------
把"非算法"部分（本文件）与"感知算法"（nut_perception.py）、"抓取规划/
控制算法"（grasp_planner.py）三者拆分成三个独立文件，便于两名同事分别
只需关注 nut_perception.py（感知）或 grasp_planner.py（抓取规划控制），
互不干扰地并行开发——两人都只需 import 本文件里的公共工具函数/常量，
不需要关心对方的算法实现细节。

本文件只包含：
  - 场景相关常量（关节名、body/site/camera 名称等）
  - 纯几何/坐标变换工具函数（不依赖任何"感知"或"规划"算法逻辑）
  - MuJoCo 机械臂/夹爪的底层读写原语（set_arm_qpos/set_arm_ctrl 等）
  - 标定/起始位姿等 json 文件加载函数
  - 3D 包围框/坐标轴的 OpenCV 可视化绘制辅助函数（纯绘图，不含任何
    位姿估计或运动规划逻辑）
不包含任何 FoundationPose 调用、IK 求解、运动队列等算法代码。
"""

import json
import math

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# ─────────────────────────────────────────────────────────────────────────────
# 场景相关常量
# ─────────────────────────────────────────────────────────────────────────────
ARM_JOINT_NAMES = ["joint_1", "joint_2", "joint_3",
                   "joint_4", "joint_5", "joint_6", "joint_7"]
# Robotiq 2F-85 由 finger_1 / finger_2 两个位置执行器直接驱动
# （right_outer_knuckle_joint 通过 xml 中的 <tendon> 与另外 4 个从动关节
# 耦合，真实物理仿真下无需手动改写全部 6 个关节 qpos）。
# 注：gen3_with_gripper_and_nuts.xml 中夹爪整体当前已被注释（未删除），
# 因此这两个执行器目前不存在，见 set_gripper_ctrl_ratio() 中的兼容处理。
GRIPPER_ACTUATOR_NAMES = ["finger_1", "finger_2"]
GRIPPER_ACTUATOR_MAX = 0.8  # 对应 xml 中 finger_1/finger_2 的 ctrlrange 上限

GRIP_SITE = "grip_site"
FLANGE_BODY_NAME = "bracelet_link"  # 末端法兰 body（夹爪已注释后，运动目标为此 body 的位置）
BASE_BODY_NAME = "base_link"
TOP_CAMERA_NAME = "d435i_top_rgb_camera"

NUT_BODY_NAMES = ["square_nut", "square_nut_2", "round_nut"]
DEFAULT_GRASP_POSE_FILES = {
    "square_nut": "nut_grasp_pose.json",
    "square_nut_2": "nut_grasp_pose.json",
    "round_nut": "nut_grasp_pose_round.json",
}

# MuJoCo 相机沿 -Z 看（OpenGL 约定）→ OpenCV 相机 +Z 朝前，相差绕 X 轴 180°
T_FLIP = np.eye(4)
T_FLIP[:3, :3] = np.diag([1.0, -1.0, -1.0])


# ─────────────────────────────────────────────────────────────────────────────
# 纯几何 / 坐标变换工具函数
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


def rotation_log(R):
    return Rotation.from_matrix(R).as_rotvec()


# ─────────────────────────────────────────────────────────────────────────────
# 机械臂 / 夹爪底层控制原语（读写 MjModel/MjData，不含任何 IK/规划逻辑）
# ─────────────────────────────────────────────────────────────────────────────

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
    关节由 xml 中的 <tendon> 被动耦合传动，无需手动写 qpos。

    注：当前 gen3_with_gripper_and_nuts.xml 中夹爪整体已被注释掉（便于
    抓取规划控制只驱动机械臂 7 个关节时不受夹爪干扰），因此
    finger_1/finger_2 执行器可能不存在（mj_name2id 返回 -1）。这里显式
    跳过缺失的执行器，避免用 -1 误写到 data.ctrl 数组最后一个元素。"""
    for name in GRIPPER_ACTUATOR_NAMES:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid == -1:
            continue
        data.ctrl[aid] = ratio * GRIPPER_ACTUATOR_MAX


# ─────────────────────────────────────────────────────────────────────────────
# 标定 / 起始位姿等 json 文件加载函数
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
    """返回该文件里的抓取候选位姿列表（4x4 矩阵的 list）。若文件里有
    "T_grasp_nut_candidates"（新格式，见 nut_grasp_pose.json 顶部注释），
    优先使用其 "candidates" 数组；否则退回旧格式，只用单个 "T_grasp_nut"
    包装成长度为 1 的列表，保持向后兼容。"""
    with open(grasp_pose_file, encoding="utf-8") as f:
        grasp = json.load(f)
    if "T_grasp_nut_candidates" in grasp:
        return [np.array(c) for c in grasp["T_grasp_nut_candidates"]["candidates"]]
    return [np.array(grasp["T_grasp_nut"])]


# ─────────────────────────────────────────────────────────────────────────────
# 轻量级 3D 框/坐标轴绘制（纯 OpenCV 绘图，不依赖 foundationpose_api 内部
# 的 torch/pytorch3d 重型导入链——那些只有真正调用 FoundationPose 推理时
# 才需要）。逻辑与 foundationpose/Utils.py 中的
# draw_posed_3d_box()/draw_xyz_axis() 完全一致（只是去掉了对
# estimater/Utils 模块的依赖），可用于画出任意目标位姿（螺母估计结果、
# 机械臂法兰目标位置等）对应的绿色 3D 框，风格与 FoundationPose 官方可视
# 化保持一致。
# ─────────────────────────────────────────────────────────────────────────────

def project_3d_to_2d(pt_homo, K, ob_in_cam):
    projected = K @ ((ob_in_cam @ pt_homo)[:3])
    projected = projected / projected[2]
    return projected[:2].round().astype(int)


def draw_posed_3d_box_simple(K, img, ob_in_cam, bbox, line_color=(0, 255, 0), linewidth=2):
    """@bbox: (2,3) min/max（局部坐标系下，与目标位姿 ob_in_cam 复合后再投影）。"""
    min_xyz = bbox.min(axis=0)
    xmin, ymin, zmin = min_xyz
    max_xyz = bbox.max(axis=0)
    xmax, ymax, zmax = max_xyz

    def draw_line3d(start, end, img):
        pts = np.stack((start, end), axis=0).reshape(-1, 3)
        pts_homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_cam = (ob_in_cam @ pts_homo.T).T[:, :3]
        projected = (K @ pts_cam.T).T
        uv = np.round(projected[:, :2] / projected[:, 2].reshape(-1, 1)).astype(int)
        img = cv2.line(img, uv[0].tolist(), uv[1].tolist(), color=line_color,
                        thickness=linewidth, lineType=cv2.LINE_AA)
        return img

    for y in [ymin, ymax]:
        for z in [zmin, zmax]:
            start = np.array([xmin, y, z])
            end = start + np.array([xmax - xmin, 0, 0])
            img = draw_line3d(start, end, img)
    for x in [xmin, xmax]:
        for z in [zmin, zmax]:
            start = np.array([x, ymin, z])
            end = start + np.array([0, ymax - ymin, 0])
            img = draw_line3d(start, end, img)
    for x in [xmin, xmax]:
        for y in [ymin, ymax]:
            start = np.array([x, y, zmin])
            end = start + np.array([0, 0, zmax - zmin])
            img = draw_line3d(start, end, img)
    return img


def draw_xyz_axis_simple(color_bgr, ob_in_cam, K, scale=0.06, thickness=2):
    """画三条从目标原点出发的坐标轴短线（BGR：X红/Y绿/Z蓝），风格与
    FoundationPose 内部 draw_xyz_axis() 一致，但只处理 BGR 图像，逻辑更简单。"""
    origin = tuple(project_3d_to_2d(np.array([0., 0., 0., 1.]), K, ob_in_cam))
    xx = tuple(project_3d_to_2d(np.array([scale, 0., 0., 1.]), K, ob_in_cam))
    yy = tuple(project_3d_to_2d(np.array([0., scale, 0., 1.]), K, ob_in_cam))
    zz = tuple(project_3d_to_2d(np.array([0., 0., scale, 1.]), K, ob_in_cam))
    cv2.arrowedLine(color_bgr, origin, xx, color=(0, 0, 255), thickness=thickness, line_type=cv2.LINE_AA)
    cv2.arrowedLine(color_bgr, origin, yy, color=(0, 255, 0), thickness=thickness, line_type=cv2.LINE_AA)
    cv2.arrowedLine(color_bgr, origin, zz, color=(255, 0, 0), thickness=thickness, line_type=cv2.LINE_AA)
    return color_bgr


# 法兰目标可视化框的半边长（米）：只是个便于观察的固定尺寸标记，与法兰的
# 真实几何尺寸无关（法兰是机械结构，不像螺母有网格可供估计包围盒）。
FLANGE_TARGET_BOX_HALF_SIZE_M = 0.03
FLANGE_TARGET_BBOX = np.array([[-FLANGE_TARGET_BOX_HALF_SIZE_M] * 3,
                                [FLANGE_TARGET_BOX_HALF_SIZE_M] * 3])

# 螺母 6D 位姿可视化框的半边长（米）：用于在画面上画出螺母位姿（不论该
# 位姿来自 FoundationPose 估计还是仿真真值）对应的简易 3D 框，同样只是个
# 便于观察的固定尺寸标记，与螺母的真实网格尺寸无关。
NUT_POSE_BOX_HALF_SIZE_M = 0.025
NUT_POSE_BBOX = np.array([[-NUT_POSE_BOX_HALF_SIZE_M] * 3,
                           [NUT_POSE_BOX_HALF_SIZE_M] * 3])

