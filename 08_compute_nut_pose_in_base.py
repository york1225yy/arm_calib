#!/usr/bin/env python3
"""
08_compute_nut_pose_in_base.py
-------------------------------
调用 foundationpose_api.py 中的 PoseEstimatorAPI，对单帧 RGBD 数据做 6D 姿态
估计，并结合机械臂运动学，把螺母（目标物体）从"相机坐标系"转换到"机械臂基
坐标系"下，最后与仿真真值（pose_estimation_data/poses/frame_000000.json）
比较，验证整条链路（手眼关系 + 正向运动学 + FoundationPose 推理）的精度。

计算链路
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
  python 08_compute_nut_pose_in_base.py \\
      --mesh nut_mesh/textured_simple.obj \\
      --data_dir pose_estimation_data --frame_idx 0 \\
      --grasp_pose_file nut_grasp_pose.json \\
      --tcp_flange_file tcp_flange.json \\
      --save_result output/nut_pose_in_base_000000.json
"""

import argparse
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
                        help="预设抓取位姿文件（T_grasp_nut：夹爪 TCP 相对螺母局部坐标系）")
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
