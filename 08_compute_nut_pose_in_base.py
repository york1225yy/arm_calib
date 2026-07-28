#!/usr/bin/env python3
"""
08_compute_nut_pose_in_base.py
-------------------------------
调用 foundationpose_api.py 中的 PoseEstimatorAPI / MultiObjectPoseEstimator，对单帧 RGBD
数据做 6D 姿态估计，并结合机械臂运动学，把目标物体从"相机坐标系"转换到
"机械臂基坐标系"下，最后与仿真真值（pose_estimation_data/poses/frame_000000.json）
比较，验证整条链路（手眼关系 + 正向运动学 + FoundationPose 推理）的精度。

多目标（双螺母、三物体等）支持
----
若 --data_dir/masks/ 下是『每个目标一个子目录』的结构（如
  pose_estimation_data_2nuts/masks/square_nut/、square_nut_2/，或
  pose_estimation_data_3nuts/masks/round_nut/、square_nut/、square_nut_2/），
本脚本会自动检测并进入多目标模式：使用 foundationpose_api.py 的
MultiObjectPoseEstimator 对每个目标分别估计，并对每个目标都输出下面描述的
完整变换链条（T_nut_cam / T_nut_base / T_grasp_base / T_flange_base 等），
并在最后额外汇总跨目标的精度统计。不同目标可能使用不同的 CAD 网格/抓取
位姿，可用 --object_mesh / --object_grasp_pose_file（格式 NAME=PATH，可
重复传入）为特定目标单独指定，未指定的目标默认用 --mesh / --grasp_pose_file。
若 --data_dir/masks/ 下直接是 *.png（旧版单螺母采集格式，如 pose_estimation_data），
则退回单目标模式（与之前行为完全一致）。

单目标模式计算链路
--------
目标在基坐标系下的位姿由三段刚体变换连乘得到：

    T_nut_base = T_gripper_base @ T_cam_gripper @ T_nut_cam

其中：

  1) T_cam_gripper（相机相对末端法兰，固定值，不随关节角变化）
     直接来自 gen3_with_nut.xml 里的运动学树结构：
       bracelet_link -> d435i_camera_body(pos=[0,-0.056,-0.060], quat=[0,1,0,0])
                      -> camera "d435i_rgb_camera"(pos=[0.0325,0,-0.0036], quat=[0,1,0,0])
     代码中不手动做四元数乘法，而是直接用 MuJoCo 在任意关节角下正向运动学算出
     bracelet_link 与相机的世界位姿，再做 T_cam_gripper = inv(T_bracelet_world) @ T_cam_world_cv。
     因为两者之间没有关节（都是固定 body 嵌套），这个结果与具体关节角无关，
     用任意 qpos 算出来都一样。

  2) T_gripper_base（末端法兰相对机械臂基座，随关节角变化）
     由 pose_estimation_data/poses/frame_000000.json 中保存的
     joint_angles_rad 通过 MuJoCo 正向运动学计算：
       T_gripper_base = inv(T_base_world) @ T_bracelet_world
     该 json 中也直接保存了同一结果（字段 gripper2base_4x4），代码里会计算
     两者并打印差异做交叉验证（应几乎为 0）。

  3) T_nut_cam（目标相对相机，6D 位姿估计结果）
     由 PoseEstimatorAPI.estimate() 对该帧的 rgb/depth/mask 做推理得到。

最终 T_nut_base 与仿真真值 T_nut_base_gt = inv(T_base_world) @ T_nut_world
做平移（mm）/旋转（度）误差对比。

在此基础上，进一步读取预设抓取位姿文件（nut_grasp_pose.json，Robotiq 2F-85
夹爪 TCP 相对螺母局部坐标系的固定抓取变换 T_grasp_nut），计算夹爪最终应到达
的抓取位姿（机械臂基坐标系下）：

    T_grasp_base = T_nut_base @ T_grasp_nut

重要说明（T_grasp_base 与真实机械臂 TCP 的关系）
----
T_grasp_base 描述的是“nut_grasp_pose.json 里设计的抓取坐标系”在基坐标系下
应该到达的位姿，它默认假设这个抓取坐标系与机械臂控制器实际配置的 TCP（工具
坐标系）完全重合。但真实场景中，Robotiq 2F-85 的 TCP（通常取两指闭合中心，
相对末端法兰 bracelet_link 有一个固定的安装偏移）需要另外通过夹爪 CAD 尺寸
或实际 TCP 标定得到，这个偏移记为：

    T_tcp_flange   （TCP 相对末端法兰 bracelet_link 的固定变换）

本脚本把它做成一个可选参数：不提供时默认为单位阵（即假设 T_grasp_base 已经
就是 TCP 目标位姿，不需要额外换算，对应本仓库当前尚未做 TCP 标定的情况）；
提供 --tcp_flange_file 后，会额外算出机械臂法兰应到达的目标位姿：

    T_flange_base = T_grasp_base @ inv(T_tcp_flange)

用法
----
  # 单目标（单螺母场景）
  python 08_compute_nut_pose_in_base.py \\
      --mesh nut_mesh/textured_simple.obj \\
      --data_dir pose_estimation_data --frame_idx 0 \\
      --grasp_pose_file nut_grasp_pose.json \\
      --tcp_flange_file tcp_flange.json \\
      --save_result output/nut_pose_in_base_000000.json

  # 多目标（三物体场景 gen3_with_two_nuts_and_round_nut.xml / pose_estimation_data_3nuts，
  # 自动检测 masks/ 下的 round_nut/、square_nut/、square_nut_2/ 三个子目录并逐个估计：
  # 两个方形螺母共用 --mesh 指定的网格/抓取位姿，round_nut 单独指定自己的网格和抓取位姿）
  python 08_compute_nut_pose_in_base.py \\
      --xml gen3_with_two_nuts_and_round_nut.xml \\
      --mesh nut_mesh/textured_simple.obj \\
      --object_mesh round_nut=nut_mesh/round_nut_textured_simple.obj \\
      --grasp_pose_file nut_grasp_pose.json \\
      --object_grasp_pose_file round_nut=nut_grasp_pose_round.json \\
      --data_dir pose_estimation_data_3nuts --frame_idx 0 \\
      --tcp_flange_file tcp_flange.json \\
      --save_result output_3nuts/nut_pose_in_base_000000.json
"""

import argparse
import glob
import json
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from foundationpose_api import PoseEstimatorAPI

# ─────────────────────────────────────────────────────────────────────────────
JOINT_NAMES = ["joint_1", "joint_2", "joint_3",
               "joint_4", "joint_5", "joint_6", "joint_7"]
CAMERA_NAME = "d435i_rgb_camera"
BASE_BODY_NAME = "base_link"
GRIPPER_BODY_NAME = "bracelet_link"

# MuJoCo 相机沿 -Z 看（OpenGL 约定）→ OpenCV 相机 +Z 朝前，相差绕 X 轴 180°
T_FLIP = np.eye(4)
T_FLIP[:3, :3] = np.diag([1.0, -1.0, -1.0])


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


def get_cam_T_world_cv(model, data, camera_name):
    """相机在世界坐标系下的位姿（OpenCV 约定：+Z 朝前）。"""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id == -1:
        raise ValueError(f"找不到相机: {camera_name}")
    T_cam_world_mj = make_T(data.cam_xpos[cam_id], data.cam_xmat[cam_id])
    return T_cam_world_mj @ T_FLIP


def apply_qpos(model, data, qpos_list):
    for name, q in zip(JOINT_NAMES, qpos_list):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        addr = model.jnt_qposadr[jid]
        data.qpos[addr] = q
    mujoco.mj_forward(model, data)


def rotation_error_deg(R1, R2):
    R_rel = R1 @ R2.T
    cos_a = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def print_T(T, indent=2):
    pad = " " * indent
    for row in T:
        print(pad + "  ".join(f"{v:10.6f}" for v in row))


# ─────────────────────────────────────────────────────────────────────────────
# 三段变换的计算
# ─────────────────────────────────────────────────────────────────────────────

def compute_T_cam_gripper(model, data):
    """相机相对末端法兰（bracelet_link）的固定位姿，与关节角无关。

    直接使用当前 data（任意关节角均可，因为 bracelet_link 与相机之间
    只有固定 body 嵌套、没有关节）算出：
        T_cam_gripper = inv(T_bracelet_world) @ T_cam_world_cv
    """
    T_bracelet_world = get_body_T_world(model, data, GRIPPER_BODY_NAME)
    T_cam_world_cv = get_cam_T_world_cv(model, data, CAMERA_NAME)
    return np.linalg.inv(T_bracelet_world) @ T_cam_world_cv


def compute_T_gripper_base(model, data, joint_angles_rad):
    """末端法兰相对机械臂基座的位姿，由关节角通过正向运动学计算。"""
    apply_qpos(model, data, joint_angles_rad)
    T_base_world = get_body_T_world(model, data, BASE_BODY_NAME)
    T_bracelet_world = get_body_T_world(model, data, GRIPPER_BODY_NAME)
    return np.linalg.inv(T_base_world) @ T_bracelet_world


def compute_T_nut_base_gt(model, data, joint_angles_rad, nut_pose_world_4x4):
    """仅用于核验：直接用仿真真值算出螺母在基坐标系下的位姿。"""
    apply_qpos(model, data, joint_angles_rad)
    T_base_world = get_body_T_world(model, data, BASE_BODY_NAME)
    T_nut_world = np.array(nut_pose_world_4x4)
    return np.linalg.inv(T_base_world) @ T_nut_world


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_grasp_pose(grasp_pose_file):
    """加载预设抓取位姿文件，返回 T_grasp_nut（夹爪 TCP 相对目标物体局部坐标系）。"""
    with open(grasp_pose_file, encoding="utf-8") as f:
        grasp = json.load(f)
    return np.array(grasp["T_grasp_nut"])


def load_tcp_flange(tcp_flange_file):
    """加载 TCP 相对末端法兰（bracelet_link）的固定变换 T_tcp_flange。

    未提供文件时返回单位阵，即默认假设 nut_grasp_pose.json 里设计的抓取坐标系
    本身就等同于机械臂控制器配置的 TCP（尚未做实际 TCP 标定/未知夹爪安装偏移
    时的保守假设）。真实使用 Robotiq 2F-85 时，应把夹爪 CAD 给出的法兰->指尖
    偏移，或实测 TCP 标定结果，填入该文件的 "T_tcp_flange" 字段。
    """
    if tcp_flange_file is None or not os.path.isfile(tcp_flange_file):
        return np.eye(4)
    with open(tcp_flange_file, encoding="utf-8") as f:
        tcp = json.load(f)
    return np.array(tcp["T_tcp_flange"])


def load_frame_inputs(data_dir, frame_idx):
    """加载指定帧的 rgb / depth / mask / 真值 json。"""
    name = f"frame_{frame_idx:06d}"

    rgb_path = os.path.join(data_dir, "rgb", f"{name}.png")
    depth_path = os.path.join(data_dir, "depth", f"{name}.png")
    mask_path = os.path.join(data_dir, "masks", f"{name}.png")
    gt_path = os.path.join(data_dir, "poses", f"{name}.json")

    bgr = cv2.imread(rgb_path)
    if bgr is None:
        raise FileNotFoundError(f"找不到 RGB 图像: {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    depth = None
    if os.path.isfile(depth_path):
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1e3  # mm -> m
        depth[(depth < 0.001) | (depth >= np.inf)] = 0

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"找不到掩码: {mask_path}")
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    return rgb, depth, mask, gt


# ─────────────────────────────────────────────────────────────────────────────
# 多目标（3 nuts 等）支持
# ─────────────────────────────────────────────────────────────────────────────

def discover_multi_object_mask_dirs(data_dir):
    """检测 <data_dir>/masks/ 是否为『每个目标一个子目录』的多目标掩码结构
    （如 pose_estimation_data_2nuts/masks/square_nut/、square_nut_2/，或
    pose_estimation_data_3nuts/masks/round_nut/、square_nut/、square_nut_2/），
    是则返回 {目标名: 子目录路径}；不是（masks/ 下直接是 *.png，如旧版单螺母
    pose_estimation_data）则返回 None，调用方应退回单目标模式。
    """
    masks_root = os.path.join(data_dir, "masks")
    if not os.path.isdir(masks_root):
        return None
    result = {}
    for entry in sorted(os.listdir(masks_root)):
        sub = os.path.join(masks_root, entry)
        if os.path.isdir(sub) and glob.glob(os.path.join(sub, "*.png")):
            result[entry] = sub
    return result or None


def parse_name_path_overrides(entries):
    """解析形如 ['NAME=PATH', ...] 的参数列表为 {名称: 路径} 字典，
    供 --object_mesh / --object_grasp_pose_file 复用。"""
    overrides = {}
    for item in entries:
        if "=" not in item:
            raise ValueError(f"格式错误（应为 NAME=PATH）: {item}")
        name, path = item.split("=", 1)
        overrides[name.strip()] = path.strip()
    return overrides


def load_frame_rgb_depth_gt(data_dir, frame_idx):
    """加载指定帧的 rgb / depth / 真值 json（不含 mask，供多目标模式使用，
    各目标的 mask 由 load_mask_for_object() 分别加载）。"""
    name = f"frame_{frame_idx:06d}"

    rgb_path = os.path.join(data_dir, "rgb", f"{name}.png")
    depth_path = os.path.join(data_dir, "depth", f"{name}.png")
    gt_path = os.path.join(data_dir, "poses", f"{name}.json")

    bgr = cv2.imread(rgb_path)
    if bgr is None:
        raise FileNotFoundError(f"找不到 RGB 图像: {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    depth = None
    if os.path.isfile(depth_path):
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1e3  # mm -> m
        depth[(depth < 0.001) | (depth >= np.inf)] = 0

    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    return rgb, depth, gt


def load_mask_for_object(mask_dir, frame_idx):
    """加载指定目标掩码子目录下某一帧的掩码，找不到/无效则返回 None
    （对应该目标在这一帧不可见的情况）。"""
    path = os.path.join(mask_dir, f"frame_{frame_idx:06d}.png")
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.max() == 0:
        return None
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def run_multi_object(args, model, data, K, mask_dirs):
    """多目标模式主流程：对 mask_dirs 中的每个目标分别执行 6D 姿态估计，
    并各自算出完整的变换链（T_nut_cam → T_nut_base → T_grasp_base →
    T_flange_base），与该目标的仿真真值比较，最后汇总跨目标的整体统计。

    与单目标模式相比：
      - 相机相对末端法兰（T_cam_gripper）、末端法兰相对基座（T_gripper_base）
        在同一帧内是所有目标共用的固定值/关节角结果，只需计算一次；
      - 每个目标各自的 mesh（--mesh / --object_mesh 覆盖）、抓取位姿文件
        （--grasp_pose_file / --object_grasp_pose_file 覆盖）、6D 位姿估计
        结果、仿真真值（gt["nuts_pose_world_4x4"][目标名]）都是独立的。
    """
    from foundationpose_api import MultiObjectPoseEstimator

    print(f"[加载数据] frame_idx={args.frame_idx} @ {args.data_dir}"
          f"（多目标模式，检测到 {len(mask_dirs)} 个目标: {list(mask_dirs.keys())}）")
    rgb, depth, gt = load_frame_rgb_depth_gt(args.data_dir, args.frame_idx)
    joint_angles_rad = gt["joint_angles_rad"]

    nuts_world_gt = gt.get("nuts_pose_world_4x4")
    if not nuts_world_gt:
        raise ValueError(
            f"{args.data_dir} 的真值 json 中没有 nuts_pose_world_4x4 字段，无法在多目标"
            f"模式下核验，请检查数据是否由支持多目标的 "
            f"05_collect_pose_estimation_data_mujoco.py 采集。"
        )

    mesh_overrides = parse_name_path_overrides(args.object_mesh)
    grasp_overrides = parse_name_path_overrides(args.object_grasp_pose_file)

    nut_names = sorted(mask_dirs.keys())
    objects = {name: mesh_overrides.get(name, args.mesh) for name in nut_names}

    print("[推理] 正在加载 MultiObjectPoseEstimator（各目标共享权重，分别加载各自网格）…")
    for name, mesh_file in objects.items():
        print(f"    {name:<16s} ← {mesh_file}")
    estimator = MultiObjectPoseEstimator(
        objects=objects,
        K=K,
        weights_dir=args.weights_dir,
        est_refine_iter=args.est_refine_iter,
        debug=0,
        debug_dir="output",
    )

    masks = {name: load_mask_for_object(mask_dirs[name], args.frame_idx) for name in nut_names}
    missing = [n for n, m in masks.items() if m is None]
    if missing:
        print(f"[警告] 以下目标在该帧没有有效掩码，将被跳过: {missing}")

    T_nut_cam_est_all = estimator.estimate_all(rgb, masks, depth=depth)

    # ── 相机相对末端法兰 / 末端法兰相对基座：同一帧内所有目标共用 ──
    T_cam_gripper = compute_T_cam_gripper(model, data)
    T_gripper_base = compute_T_gripper_base(model, data, joint_angles_rad)
    if "gripper2base_4x4" in gt:
        T_gripper_base_saved = np.array(gt["gripper2base_4x4"])
        diff = np.abs(T_gripper_base - T_gripper_base_saved).max()
        print(f"[核验] 正向运动学计算 vs json 保存的 gripper2base_4x4，最大元素差异 = {diff:.3e}")

    T_tcp_flange = load_tcp_flange(args.tcp_flange_file)
    tcp_flange_desc = (os.path.basename(args.tcp_flange_file)
                       if args.tcp_flange_file else "未提供，默认单位阵")

    per_object_results = {}
    trans_errs, rot_errs = [], []
    sep = "=" * 65

    for name in nut_names:
        if name not in T_nut_cam_est_all:
            continue  # 该帧该目标无有效掩码，跳过

        T_nut_cam_est = T_nut_cam_est_all[name]
        T_nut_base_est = T_gripper_base @ T_cam_gripper @ T_nut_cam_est
        T_nut_base_gt = compute_T_nut_base_gt(model, data, joint_angles_rad, nuts_world_gt[name])

        grasp_file = grasp_overrides.get(name, args.grasp_pose_file)
        T_grasp_nut = load_grasp_pose(grasp_file)
        T_grasp_base_est = T_nut_base_est @ T_grasp_nut
        T_flange_base_est = T_grasp_base_est @ np.linalg.inv(T_tcp_flange)

        R_est, t_est = T_nut_base_est[:3, :3], T_nut_base_est[:3, 3]
        R_gt, t_gt = T_nut_base_gt[:3, :3], T_nut_base_gt[:3, 3]
        trans_err_mm = float(np.linalg.norm(t_est - t_gt) * 1000)
        rot_err_deg = float(rotation_error_deg(R_est, R_gt))
        trans_errs.append(trans_err_mm)
        rot_errs.append(rot_err_deg)

        print(f"\n{sep}")
        print(f"  帧 {args.frame_idx:06d} · 目标 '{name}'：在机械臂基坐标系下的位姿")
        print(sep)
        print("\n[T_cam_gripper  相机 -> 末端法兰（固定值，同一帧内各目标共用）]")
        print_T(T_cam_gripper)
        print("\n[T_gripper_base 末端法兰 -> 基座（随关节角变化，同一帧内各目标共用）]")
        print_T(T_gripper_base)
        print(f"\n[T_nut_cam      {name} -> 相机（FoundationPose 估计）]")
        print_T(T_nut_cam_est)
        print(f"\n[T_nut_base_est {name} -> 基座（估计结果，三段连乘）]")
        print_T(T_nut_base_est)
        print(f"\n[T_nut_base_gt  {name} -> 基座（仿真真值，仅供核验）]")
        print_T(T_nut_base_gt)
        print(f"\n  平移误差 ||Δt|| = {trans_err_mm:.4f} mm")
        print(f"  旋转误差 angle  = {rot_err_deg:.4f} °")
        print(f"\n[T_grasp_nut    夹爪 TCP -> {name}（预设抓取位姿，来自 {os.path.basename(grasp_file)}）]")
        print_T(T_grasp_nut)
        print(f"\n[T_grasp_base_est 夹爪 TCP -> 基座（{name} 的最终抓取目标位姿）]")
        print_T(T_grasp_base_est)
        print(f"\n[T_tcp_flange   TCP -> 末端法兰（{tcp_flange_desc}）]")
        print_T(T_tcp_flange)
        print(f"\n[T_flange_base_est 末端法兰 -> 基座（{name} 对应的法兰目标位姿）]")
        print_T(T_flange_base_est)
        print(sep)

        per_object_results[name] = {
            "T_cam_gripper": T_cam_gripper.tolist(),
            "T_gripper_base": T_gripper_base.tolist(),
            "T_nut_cam_est": T_nut_cam_est.tolist(),
            "T_nut_base_est": T_nut_base_est.tolist(),
            "T_nut_base_gt": T_nut_base_gt.tolist(),
            "translation_error_mm": trans_err_mm,
            "rotation_error_deg": rot_err_deg,
            "mesh_file": objects[name],
            "grasp_pose_file": grasp_file,
            "T_grasp_nut": T_grasp_nut.tolist(),
            "T_grasp_base_est": T_grasp_base_est.tolist(),
            "T_tcp_flange": T_tcp_flange.tolist(),
            "T_flange_base_est": T_flange_base_est.tolist(),
        }

    if trans_errs:
        ta, ra = np.array(trans_errs), np.array(rot_errs)
        print(f"\n{sep}")
        print(f"  全部 {len(trans_errs)} 个目标合计统计")
        print(sep)
        print(f"  {'':>10}  {'旋转误差 (°)':>14}  {'平移误差 (mm)':>16}")
        print(f"  {'均值':>10}  {ra.mean():>14.4f}  {ta.mean():>16.4f}")
        print(f"  {'最大值':>10}  {ra.max():>14.4f}  {ta.max():>16.4f}")
        print(sep)
    else:
        print("[警告] 该帧没有任何目标估计出有效位姿。")

    if args.save_result:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_result)), exist_ok=True)
        result = {
            "frame_idx": args.frame_idx,
            "mode": "multi_object",
            "objects": per_object_results,
        }
        if trans_errs:
            result["overall_summary"] = {
                "n_objects": len(trans_errs),
                "translation_error_mm": {"mean": float(np.mean(trans_errs)), "max": float(np.max(trans_errs))},
                "rotation_error_deg": {"mean": float(np.mean(rot_errs)), "max": float(np.max(rot_errs))},
            }
        with open(args.save_result, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[保存] 结果已保存至 {args.save_result}")


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="用 PoseEstimatorAPI 估计螺母位姿，再结合机械臂运动学换算到基坐标系，并与仿真真值比较",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xml", default=os.path.join(here, "gen3_with_nut.xml"))
    parser.add_argument("--mesh", default=os.path.join(here, "nut_mesh", "textured_simple.obj"))
    parser.add_argument("--weights_dir", default=os.path.join(here, "weights"))
    parser.add_argument("--data_dir", default=os.path.join(here, "pose_estimation_data"))
    parser.add_argument("--cam_K_file", default=None,
                        help="相机内参文件路径，默认取 <data_dir>/cam_K.txt")
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--grasp_pose_file", default=os.path.join(here, "nut_grasp_pose.json"),
                        help="预设抓取位姿文件（T_grasp_nut：夹爪 TCP 相对螺母局部坐标系）"
                             "。多目标模式下作为未被 --object_grasp_pose_file 单独覆盖的目标的默认值。")
    parser.add_argument("--object_mesh", action="append", default=[],
                        metavar="NAME=PATH",
                        help="仅多目标模式下使用：为指定目标单独指定 mesh 文件（可重复传入），"
                             "格式 'NAME=PATH'，例如 --object_mesh round_nut=nut_mesh/round_nut_textured_simple.obj。"
                             "未覆盖的目标默认使用 --mesh（适用于两个方形螺母共用同一个网格的情况）。")
    parser.add_argument("--object_grasp_pose_file", action="append", default=[],
                        metavar="NAME=PATH",
                        help="仅多目标模式下使用：为指定目标单独指定抓取位姿文件（可重复传入），"
                             "格式 'NAME=PATH'，例如 --object_grasp_pose_file round_nut=nut_grasp_pose_round.json。"
                             "未覆盖的目标默认使用 --grasp_pose_file。")
    parser.add_argument("--tcp_flange_file", default=None,
                        help="可选：TCP 相对末端法兰的标定/CAD 文件（字段 T_tcp_flange）。"
                             "不提供则默认单位阵，即假设抓取坐标系本身就是 TCP。")
    parser.add_argument("--save_result", default=None,
                        help="将结果保存为 json（如 output/nut_pose_in_base_000000.json）")
    args = parser.parse_args()

    cam_K_file = args.cam_K_file or os.path.join(args.data_dir, "cam_K.txt")
    K = np.loadtxt(cam_K_file).reshape(3, 3)

    print(f"[加载模型] {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    # MjData 刚创建时 xpos/xmat 均为 0（尚未做过正向运动学），
    # 必须先跑一次 mj_forward 填充有效的世界位姿，否则后续
    # compute_T_cam_gripper 里读到的 body 位姿是全零矩阵（奇异矩阵）。
    mujoco.mj_forward(model, data)

    # ── 多目标模式检测：<data_dir>/masks/ 下是否为『每个目标一个子目录』的结构 ──
    mask_dirs = discover_multi_object_mask_dirs(args.data_dir)
    if mask_dirs:
        run_multi_object(args, model, data, K, mask_dirs)
        return

    print(f"[加载数据] frame_idx={args.frame_idx} @ {args.data_dir}")
    rgb, depth, mask, gt = load_frame_inputs(args.data_dir, args.frame_idx)
    joint_angles_rad = gt["joint_angles_rad"]

    # ── 1) 相机相对末端法兰（固定值，与关节角无关）──
    T_cam_gripper = compute_T_cam_gripper(model, data)

    # ── 2) 末端法兰相对基座（由关节角正向运动学计算）──
    T_gripper_base = compute_T_gripper_base(model, data, joint_angles_rad)

    # 与 json 中保存的 gripper2base_4x4 交叉验证（理论上应几乎一致）
    if "gripper2base_4x4" in gt:
        T_gripper_base_saved = np.array(gt["gripper2base_4x4"])
        diff = np.abs(T_gripper_base - T_gripper_base_saved).max()
        print(f"[核验] 正向运动学计算 vs json 保存的 gripper2base_4x4，最大元素差异 = {diff:.3e}")

    # ── 3) 目标相对相机：调用 FoundationPose API 推理 ──
    print("[推理] 正在加载 PoseEstimatorAPI 并执行姿态估计 …")
    estimator = PoseEstimatorAPI(
        mesh_file=args.mesh,
        K=K,
        weights_dir=args.weights_dir,
        est_refine_iter=args.est_refine_iter,
        debug=0,
        debug_dir="output",
    )
    T_nut_cam_est = estimator.estimate(rgb=rgb, mask=mask, depth=depth)

    # ── 三段变换连乘：目标在基坐标系下的位姿 ──
    T_nut_base_est = T_gripper_base @ T_cam_gripper @ T_nut_cam_est

    # ── 仿真真值（仅用于核验，实际部署时不存在）──
    T_nut_base_gt = compute_T_nut_base_gt(
        model, data, joint_angles_rad, gt["nut_pose_world_4x4"]
    )

    # ── 4) 结合预设抓取位姿：夹爪最终应到达的抓取位姿（基坐标系下）──
    T_grasp_nut = load_grasp_pose(args.grasp_pose_file)
    T_grasp_base_est = T_nut_base_est @ T_grasp_nut

    # ── 5) TCP 相对末端法兰的固定偏移（默认单位阵，可填入真实 CAD/标定值）──
    #      T_flange_base = T_grasp_base @ inv(T_tcp_flange)：机械臂法兰应到达的目标位姿
    T_tcp_flange = load_tcp_flange(args.tcp_flange_file)
    T_flange_base_est = T_grasp_base_est @ np.linalg.inv(T_tcp_flange)

    R_est, t_est = T_nut_base_est[:3, :3], T_nut_base_est[:3, 3]
    R_gt, t_gt = T_nut_base_gt[:3, :3], T_nut_base_gt[:3, 3]
    trans_err_mm = float(np.linalg.norm(t_est - t_gt) * 1000)
    rot_err_deg = float(rotation_error_deg(R_est, R_gt))

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  帧 {args.frame_idx:06d}：螺母在机械臂基坐标系下的位姿")
    print(sep)
    print("\n[T_cam_gripper  相机 -> 末端法兰（固定值）]")
    print_T(T_cam_gripper)
    print("\n[T_gripper_base 末端法兰 -> 基座（随关节角变化）]")
    print_T(T_gripper_base)
    print("\n[T_nut_cam      螺母 -> 相机（FoundationPose 估计）]")
    print_T(T_nut_cam_est)
    print("\n[T_nut_base_est 螺母 -> 基座（估计结果，三段连乘）]")
    print_T(T_nut_base_est)
    print("\n[T_nut_base_gt  螺母 -> 基座（仿真真值，仅供核验）]")
    print_T(T_nut_base_gt)
    print(f"\n  平移误差 ||Δt|| = {trans_err_mm:.4f} mm")
    print(f"  旋转误差 angle  = {rot_err_deg:.4f} °")
    print(f"\n[T_grasp_nut    夹爪 TCP -> 螺母（预设抓取位姿，来自 {os.path.basename(args.grasp_pose_file)}）]")
    print_T(T_grasp_nut)
    print("\n[T_grasp_base_est 夹爪 TCP -> 基座（最终抓取目标位姿，假设抓取坐标系即 TCP）]")
    print_T(T_grasp_base_est)
    tcp_flange_desc = (os.path.basename(args.tcp_flange_file)
                       if args.tcp_flange_file else "未提供，默认单位阵")
    print(f"\n[T_tcp_flange   TCP -> 末端法兰（{tcp_flange_desc}）]")
    print_T(T_tcp_flange)
    print("\n[T_flange_base_est 末端法兰 -> 基座（真正下发给机械臂控制器的法兰目标位姿）]")
    print_T(T_flange_base_est)
    print(sep)

    if args.save_result:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_result)), exist_ok=True)
        result = {
            "frame_idx": args.frame_idx,
            "T_cam_gripper": T_cam_gripper.tolist(),
            "T_gripper_base": T_gripper_base.tolist(),
            "T_nut_cam_est": T_nut_cam_est.tolist(),
            "T_nut_base_est": T_nut_base_est.tolist(),
            "T_nut_base_gt": T_nut_base_gt.tolist(),
            "translation_error_mm": trans_err_mm,
            "rotation_error_deg": rot_err_deg,
            "T_grasp_nut": T_grasp_nut.tolist(),
            "T_grasp_base_est": T_grasp_base_est.tolist(),
            "T_tcp_flange": T_tcp_flange.tolist(),
            "T_flange_base_est": T_flange_base_est.tolist(),
        }
        with open(args.save_result, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[保存] 结果已保存至 {args.save_result}")


if __name__ == "__main__":
    main()
