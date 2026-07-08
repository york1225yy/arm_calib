#!/usr/bin/env python3
"""
04_verify_calibration.py
-------------------------
通过对比手眼标定结果与 MuJoCo 建模真值，定量验证标定精度。

原理
----
在仿真环境中，相机安装在机械臂末端的几何关系已知（由 XML 建模定义）。
MuJoCo 正向运动学可在任意关节角下精确给出：
  T_cam_world    = cam_xpos / cam_xmat
  T_gripper_world = xpos / xmat (bracelet_link)
  T_cam2gripper_GT = inv(T_gripper_world) @ T_cam_world

标定结果 T_cam2end 来自 hand_eye_result_eye_in_hand.json。

两者比较即可得到"绝对误差"（vs 只能算相对一致性的 AX=XB 误差）。

用法
----
  python 04_verify_calibration.py
  python 04_verify_calibration.py --result calibration_data/hand_eye_result_eye_in_hand.json
  python 04_verify_calibration.py --xml gen3_with_camera.xml --poses calibration_data/poses
"""

import argparse
import glob
import json
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# ─────────────────────────────────────────────────────────────────────────────
JOINT_NAMES  = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7"]
HOME_QPOS    = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
CAMERA_NAME  = "d435i_rgb_camera"
GRIPPER_NAME = "bracelet_link"


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def make_T(pos, mat):
    T = np.eye(4)
    T[:3, :3] = np.array(mat).reshape(3, 3)
    T[:3,  3] = np.array(pos)
    return T


def apply_qpos(model, data, qpos_list):
    for name, q in zip(JOINT_NAMES, qpos_list):
        jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        addr = model.jnt_qposadr[jid]
        data.qpos[addr] = q
    mujoco.mj_forward(model, data)


def get_T_cam2gripper_gt(model, data):
    """从 MuJoCo 运动学精确计算相机→末端（bracelet_link）变换（Ground Truth）。"""
    cam_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   GRIPPER_NAME)

    T_cam     = make_T(data.cam_xpos[cam_id],     data.cam_xmat[cam_id])
    T_gripper = make_T(data.xpos[gripper_id],     data.xmat[gripper_id])
    return np.linalg.inv(T_gripper) @ T_cam


def rotation_error_deg(R1, R2):
    R_rel = R1 @ R2.T
    cos_a = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def print_T(T, indent=2):
    pad = " " * indent
    for row in T:
        print(pad + "  ".join(f"{v:10.6f}" for v in row))


def quality_grade(rot_deg, trans_mm):
    if rot_deg < 0.5 and trans_mm < 2.0:
        return "EXCELLENT", "标定结果与建模真值高度吻合，误差处于数值精度级别。"
    if rot_deg < 1.0 and trans_mm < 5.0:
        return "GOOD",      "标定结果良好，满足大多数工业应用需求。"
    if rot_deg < 3.0 and trans_mm < 15.0:
        return "ACCEPTABLE","标定结果尚可，建议增加位姿多样性（J1 大范围旋转）后重新标定。"
    return "POOR",          "标定偏差较大，请参考 03_hand_eye_calibration.py 的建议重新采集。"


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="手眼标定结果 vs MuJoCo 建模真值 对比验证",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xml",    default=os.path.join(os.path.dirname(__file__), "gen3_with_camera.xml"))
    parser.add_argument("--result", default=os.path.join(os.path.dirname(__file__), "calibration_data/hand_eye_result_eye_in_hand.json"))
    parser.add_argument("--poses",  default=os.path.join(os.path.dirname(__file__), "calibration_data/poses"),
                        help="保存的位姿目录，用于多位姿一致性验证（可选）")
    args = parser.parse_args()

    sep  = "=" * 65
    sep2 = "-" * 65

    # ── 加载标定结果 ──────────────────────────────────────────────────────────
    print(f"\n[加载标定结果] {args.result}")
    with open(args.result, encoding="utf-8") as f:
        cal = json.load(f)
    T_cal = np.array(cal["T_cam2end_4x4"])
    R_cal = T_cal[:3, :3]
    t_cal = T_cal[:3,  3]
    print(f"  算法: {cal.get('method','?')}   有效位姿对: {cal.get('n_valid_pairs','?')}")
    print(f"  AX=XB 一致性误差（来自 03 脚本）: "
          f"旋转 {cal['accuracy']['consistency_rotation_error_mean_deg']:.4f}°  "
          f"平移 {cal['accuracy']['consistency_translation_error_mean_mm']:.4f} mm")

    # ── 加载 MuJoCo 模型 ──────────────────────────────────────────────────────
    print(f"\n[加载模型] {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)

    # ── 计算 Ground Truth ─────────────────────────────────────────────────────
    # T_cam2gripper 只取决于刚体安装关系，与关节角无关；
    # 用 home 姿态计算即可，多个姿态结果应完全一致（数值误差 <1e-10）。
    apply_qpos(model, data, HOME_QPOS)
    T_gt = get_T_cam2gripper_gt(model, data)
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3,  3]

    # 验证真值与姿态无关（用第一个保存位姿再算一次）
    pose_files = sorted(glob.glob(os.path.join(args.poses, "calib_pose_*.json")))
    if pose_files:
        pd = json.load(open(pose_files[0], encoding="utf-8"))
        apply_qpos(model, data, pd["joint_angles_rad"])
        T_gt2 = get_T_cam2gripper_gt(model, data)
        gt_stability = np.linalg.norm(T_gt[:3, 3] - T_gt2[:3, 3]) * 1000
        print(f"\n[真值稳定性验证] 两姿态下真值平移差 = {gt_stability:.2e} mm  "
              f"{'✓' if gt_stability < 0.01 else '⚠ 异常'}")
        apply_qpos(model, data, HOME_QPOS)  # 恢复

    # ── 坐标系约定转换 ────────────────────────────────────────────────────────
    # MuJoCo 相机沿 -Z 方向观看（OpenGL 约定），cam_xmat 给出的相机帧中 -Z = 视线方向。
    # OpenCV / solvePnP 的相机帧定义为 +Z 朝前、+X 右、+Y 下。
    # 两者相差一个绕 X 轴 180° 的旋转：R_flip = diag(1,-1,-1)。
    # 因此要将 MuJoCo 的 T_cam2gripper_GT 转换到 OpenCV 帧再与标定结果比较：
    #   T_gt_opencv = T_gt_mujoco @ [[R_flip, 0], [0, 1]]
    R_flip = np.diag([1.0, -1.0, -1.0])
    T_flip = np.eye(4); T_flip[:3, :3] = R_flip
    T_gt_opencv = T_gt @ T_flip
    R_gt_opencv = T_gt_opencv[:3, :3]
    t_gt_opencv = T_gt_opencv[:3,  3]

    # ── 误差计算 ──────────────────────────────────────────────────────────────
    rot_err_deg  = rotation_error_deg(R_cal, R_gt_opencv)
    trans_err_mm = np.linalg.norm(t_cal - t_gt_opencv) * 1000
    euler_cal    = Rotation.from_matrix(R_cal      ).as_euler("XYZ", degrees=True)
    euler_gt     = Rotation.from_matrix(R_gt_opencv).as_euler("XYZ", degrees=True)

    # ── 打印结果 ──────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  手眼标定结果 vs MuJoCo 建模真值（Ground Truth）")
    print(sep)

    print("\n[标定结果  T_cam→gripper  (OpenCV frame: +Z forward)]")
    print_T(T_cal)
    print("\n[建模真值  T_cam→gripper  (MuJoCo raw: -Z forward)]")
    print_T(T_gt)
    print("\n[建模真值  T_cam→gripper  (转换为 OpenCV frame 后)]")
    print_T(T_gt_opencv)

    print(f"\n{sep2}")
    print("  平移分量对比（OpenCV frame）")
    print(sep2)
    print(f"  {'轴':>4}  {'标定 (mm)':>12}  {'真值 (mm)':>12}  {'差值 (mm)':>12}")
    for ax, tc, tg in zip(["X", "Y", "Z"], t_cal * 1000, t_gt_opencv * 1000):
        print(f"  {ax:>4}  {tc:>12.3f}  {tg:>12.3f}  {tc - tg:>+12.3f}")
    print(f"\n  平移总误差 ||Δt|| = {trans_err_mm:.4f} mm")

    print(f"\n{sep2}")
    print("  旋转分量对比（XYZ Euler，度）（OpenCV frame）")
    print(sep2)
    print(f"  {'轴':>4}  {'标定 (°)':>12}  {'真值 (°)':>12}  {'差值 (°)':>12}")
    for ax, ec, eg in zip(["X", "Y", "Z"], euler_cal, euler_gt):
        diff = ec - eg
        if diff >  180: diff -= 360
        if diff < -180: diff += 360
        print(f"  {ax:>4}  {ec:>12.4f}  {eg:>12.4f}  {diff:>+12.4f}")
    print(f"\n  旋转总误差 angle(R_cal·R_gt^T) = {rot_err_deg:.4f} °")

    grade, tip = quality_grade(rot_err_deg, trans_err_mm)
    print(f"\n{sep}")
    print(f"  综合质量等级: {grade}")
    print(f"  {tip}")
    print(sep)

    # ── 多位姿验证：标定后相机位置的绝对误差 ────────────────────────────────
    if pose_files:
        print(f"\n[多位姿验证] 对 {len(pose_files)} 个保存位姿，计算"
              "使用标定结果 vs 真值 在世界系中的相机位置误差")
        print(f"  {'位姿文件':>25}  {'位置误差 (mm)':>16}  {'旋转误差 (°)':>14}")
        pos_errs, rot_errs = [], []
        gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_NAME)
        cam_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        for pf in pose_files:
            pd = json.load(open(pf, encoding="utf-8"))
            apply_qpos(model, data, pd["joint_angles_rad"])

            T_grip_w = make_T(data.xpos[gripper_id], data.xmat[gripper_id])

            # 标定估计：用 T_cal（OpenCV frame）推算相机在世界系位置
            T_cam_w_cal = T_grip_w @ T_cal
            # 真值：MuJoCo 直接读取，需转换为 OpenCV frame
            T_cam_w_gt_raw = make_T(data.cam_xpos[cam_id], data.cam_xmat[cam_id])
            T_cam_w_gt     = T_cam_w_gt_raw @ T_flip

            pe = np.linalg.norm(T_cam_w_cal[:3, 3] - T_cam_w_gt[:3, 3]) * 1000
            re = rotation_error_deg(T_cam_w_cal[:3, :3], T_cam_w_gt[:3, :3])
            pos_errs.append(pe)
            rot_errs.append(re)
            print(f"  {os.path.basename(pf):>25}  {pe:>16.4f}  {re:>14.4f}")

        pa = np.array(pos_errs)
        ra = np.array(rot_errs)
        print(f"\n  {'均值':>25}  {pa.mean():>16.4f}  {ra.mean():>14.4f}")
        print(f"  {'最大值':>25}  {pa.max():>16.4f}  {ra.max():>14.4f}")
        print(f"  {'标准差':>25}  {pa.std():>16.4f}  {ra.std():>14.4f}")
        print(sep)


if __name__ == "__main__":
    main()
