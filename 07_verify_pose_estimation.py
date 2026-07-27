#!/usr/bin/env python3
"""
07_verify_pose_estimation.py
-------------------------------
对比 FoundationPose 姿态估计结果与 MuJoCo 仿真真值，定量验证 6D 位姿估计精度。

原理
----
FoundationPose（foundationpose_api.py）对每一帧输出的是物体（螺母网格）
在**相机坐标系**（OpenCV 约定：+X 右, +Y 下, +Z 前）下的位姿：

    T_nut_cam_est   (output/poses/000000.txt 等)

而 MuJoCo 仿真在采集数据时（05_collect_pose_estimation_data_mujoco.py）
针对每一帧保存了：
    - joint_angles_rad     ：采集时刻各关节角
    - nut_pose_world_4x4   ：第一个螺母（square_nut）在世界坐标系下的真值位姿（兼容旧版单螺母脚本）
    - nuts_pose_world_4x4  ：{螺母名: 世界坐标系真值位姿} 字典，包含场景中全部螺母
                             （如 gen3_with_two_nuts.xml 场景中的 "square_nut"/"square_nut_2"）

结合 gen3_with_nut.xml 建模参数，用 MuJoCo 正向运动学可以精确计算出相机
在世界坐标系下的位姿 T_cam_world（MuJoCo 原始约定：相机沿 -Z 看向目标），
经过一次绕 X 轴 180° 的坐标系翻转即可转换为 OpenCV 约定：

    T_cam_world_cv = T_cam_world_mj @ diag(1, -1, -1, 1)

进而得到螺母在相机坐标系下的真值：

    T_nut_cam_gt = inv(T_cam_world_cv) @ T_nut_world

将 T_nut_cam_est 与 T_nut_cam_gt 对比，即可得到姿态估计的绝对误差
（平移 mm + 旋转 度）。

注意：06_convert_nut_mesh_for_foundationpose.py 生成网格时刻意让网格局部
坐标系与 gen3_with_nut.xml 中 body "square_nut" 的坐标系完全对齐，所以此
处无需额外做网格原点/朝向的换算（场景中的其余螺母如 "square_nut_2" 使用
完全相同的几何体/网格，仅世界位姿不同，同样无需额外换算）。

多目标（双螺母等）支持
----
若 foundationpose_api.py 是以多目标模式运行的（见其 MultiObjectPoseEstimator /
--mask_dir 多子目录用法），估计结果会分别保存在 --est_dir 下的多个子目录中
（如 output_2nuts/poses/square_nut/、output_2nuts/poses/square_nut_2/）。本脚本
会自动检测这种『每个目标一个子目录』的结构并进入多目标批量对比模式：对每个
子目录分别按帧号与 --gt_dir 中 json 的 nuts_pose_world_4x4[对应目标名] 配对
比较，并额外打印/保存跨目标的整体汇总统计。单帧模式下也可用 --nut_name 指定
要对比的目标名（对应 nuts_pose_world_4x4 的 key）。

用法
----
  # 对比单帧（单螺母场景）
  python 07_verify_pose_estimation.py \\
      --est_pose output/poses/000000.txt \\
      --gt_pose  pose_estimation_data/poses/frame_000000.json

  # 对比单帧（双螺母场景，指定要核验的目标名）
  python 07_verify_pose_estimation.py \\
      --est_pose output_2nuts/poses/square_nut_2/000000.txt \\
      --gt_pose  pose_estimation_data_2nuts/poses/frame_000000.json \\
      --nut_name square_nut_2

  # 批量对比目录下所有匹配的帧（按序号自动配对，单目标）
  python 07_verify_pose_estimation.py \\
      --est_dir output/poses --gt_dir pose_estimation_data/poses

  # 批量对比（双螺母场景，自动检测 est_dir 下的 square_nut/、square_nut_2/ 子目录）
  python 07_verify_pose_estimation.py \\
      --est_dir output_2nuts/poses --gt_dir pose_estimation_data_2nuts/poses \\
      --save_result output_2nuts/pose_verify_result.json
"""

import argparse
import glob
import json
import math
import os
import re

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# ─────────────────────────────────────────────────────────────────────────────
JOINT_NAMES = ["joint_1", "joint_2", "joint_3",
               "joint_4", "joint_5", "joint_6", "joint_7"]
CAMERA_NAME = "d435i_rgb_camera"

# MuJoCo 相机沿 -Z 看（OpenGL 约定） → OpenCV 相机 +Z 朝前，相差绕 X 轴 180°
R_FLIP = np.diag([1.0, -1.0, -1.0])
T_FLIP = np.eye(4)
T_FLIP[:3, :3] = R_FLIP


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def make_T(pos, mat):
    T = np.eye(4)
    T[:3, :3] = np.array(mat).reshape(3, 3)
    T[:3, 3] = np.array(pos)
    return T


def apply_qpos(model, data, qpos_list):
    for name, q in zip(JOINT_NAMES, qpos_list):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        addr = model.jnt_qposadr[jid]
        data.qpos[addr] = q
    mujoco.mj_forward(model, data)


def get_T_cam_world_cv(model, data, camera_name=CAMERA_NAME):
    """计算相机在世界坐标系下的位姿（OpenCV 约定：+Z 朝前）。"""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    T_cam_world_mj = make_T(data.cam_xpos[cam_id], data.cam_xmat[cam_id])
    return T_cam_world_mj @ T_FLIP


def rotation_error_deg(R1, R2):
    R_rel = R1 @ R2.T
    cos_a = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def print_T(T, indent=2):
    pad = " " * indent
    for row in T:
        print(pad + "  ".join(f"{v:10.6f}" for v in row))


def quality_grade(rot_deg, trans_mm):
    if rot_deg < 2.0 and trans_mm < 5.0:
        return "EXCELLENT", "位姿估计精度很高，可直接用于精细抓取。"
    if rot_deg < 5.0 and trans_mm < 10.0:
        return "GOOD", "位姿估计精度良好，满足大多数抓取应用需求。"
    if rot_deg < 10.0 and trans_mm < 20.0:
        return "ACCEPTABLE", "位姿估计尚可，建议提高 mask 精度或增大 --est_refine_iter 后重试。"
    return "POOR", "位姿估计偏差较大，请检查 mask 质量、深度图有效性及网格坐标系是否对齐。"


def load_gt_nut_in_cam(model, data, gt_json_path, nut_name=None):
    """加载单帧真值 json，返回 (T_nut_cam_gt, joint_angles)。

    参数
    ----
    nut_name : 可选，指定读取哪个螺母的真值，对应
        05_collect_pose_estimation_data_mujoco.py 写入的
        nuts_pose_world_4x4 字典的 key（如 "square_nut"/"square_nut_2"）。
        - 提供时：必须能在 nuts_pose_world_4x4 中找到，找不到则报错。
        - 不提供时（默认）：优先使用旧版单螺母字段 nut_pose_world_4x4
          （向后兼容单螺母场景采集的数据）；该字段不存在时退回
          nuts_pose_world_4x4 中的第一个螺母。
    """
    with open(gt_json_path, encoding="utf-8") as f:
        gt = json.load(f)
    apply_qpos(model, data, gt["joint_angles_rad"])
    T_cam_world_cv = get_T_cam_world_cv(model, data)

    if nut_name is not None:
        nuts = gt.get("nuts_pose_world_4x4")
        if not nuts or nut_name not in nuts:
            raise KeyError(
                f"真值 json 中找不到螺母 '{nut_name}' 的位姿（{gt_json_path}），"
                f"可用的螺母名: {list((nuts or {}).keys())}"
            )
        T_nut_world = np.array(nuts[nut_name])
    elif "nut_pose_world_4x4" in gt:
        T_nut_world = np.array(gt["nut_pose_world_4x4"])
    else:
        nuts = gt["nuts_pose_world_4x4"]
        T_nut_world = np.array(next(iter(nuts.values())))

    T_nut_cam_gt = np.linalg.inv(T_cam_world_cv) @ T_nut_world
    return T_nut_cam_gt, gt["joint_angles_rad"]


def load_est_pose(est_txt_path):
    T = np.loadtxt(est_txt_path).reshape(4, 4)
    return T


def compare_one(model, data, est_path, gt_path, nut_name=None, verbose=True):
    T_gt, _ = load_gt_nut_in_cam(model, data, gt_path, nut_name=nut_name)
    T_est = load_est_pose(est_path)

    R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]
    R_est, t_est = T_est[:3, :3], T_est[:3, 3]

    rot_err_deg = rotation_error_deg(R_est, R_gt)
    trans_err_mm = np.linalg.norm(t_est - t_gt) * 1000
    euler_est = Rotation.from_matrix(R_est).as_euler("XYZ", degrees=True)
    euler_gt = Rotation.from_matrix(R_gt).as_euler("XYZ", degrees=True)

    grade, tip = quality_grade(rot_err_deg, trans_err_mm)
    result = {
        "nut_name": nut_name,
        "est_file": os.path.basename(est_path),
        "gt_file": os.path.basename(gt_path),
        "T_est_nut2cam": T_est.tolist(),
        "T_gt_nut2cam": T_gt.tolist(),
        "translation_mm": {
            "est": (t_est * 1000).tolist(),
            "gt": (t_gt * 1000).tolist(),
            "diff": ((t_est - t_gt) * 1000).tolist(),
            "error_norm_mm": float(trans_err_mm),
        },
        "rotation_euler_xyz_deg": {
            "est": euler_est.tolist(),
            "gt": euler_gt.tolist(),
        },
        "rotation_error_deg": float(rot_err_deg),
        "quality_grade": grade,
        "quality_tip": tip,
    }

    if verbose:
        sep, sep2 = "=" * 65, "-" * 65
        label = f"  [螺母: {nut_name}]" if nut_name else ""
        print(f"\n{sep}")
        print(f"  帧对比{label}: {os.path.basename(est_path)}  vs  {os.path.basename(gt_path)}")
        print(sep)

        print("\n[FoundationPose 估计  T_nut→cam  (OpenCV frame)]")
        print_T(T_est)
        print("\n[MuJoCo 建模真值      T_nut→cam  (OpenCV frame)]")
        print_T(T_gt)

        print(f"\n{sep2}")
        print("  平移分量对比")
        print(sep2)
        print(f"  {'轴':>4}  {'估计 (mm)':>12}  {'真值 (mm)':>12}  {'差值 (mm)':>12}")
        for ax, tc, tg in zip(["X", "Y", "Z"], t_est * 1000, t_gt * 1000):
            print(f"  {ax:>4}  {tc:>12.3f}  {tg:>12.3f}  {tc - tg:>+12.3f}")
        print(f"\n  平移总误差 ||Δt|| = {trans_err_mm:.4f} mm")

        print(f"\n{sep2}")
        print("  旋转分量对比（XYZ Euler，度）")
        print(sep2)
        print(f"  {'轴':>4}  {'估计 (°)':>12}  {'真值 (°)':>12}  {'差值 (°)':>12}")
        for ax, ec, eg in zip(["X", "Y", "Z"], euler_est, euler_gt):
            diff = ec - eg
            if diff > 180:
                diff -= 360
            if diff < -180:
                diff += 360
            print(f"  {ax:>4}  {ec:>12.4f}  {eg:>12.4f}  {diff:>+12.4f}")
        print(f"\n  旋转总误差 angle(R_est·R_gt^T) = {rot_err_deg:.4f} °")

        grade, tip = quality_grade(rot_err_deg, trans_err_mm)
        print(f"\n  综合质量等级: {grade}  —— {tip}")
        print(sep)

    return result


def _frame_index(path):
    m = re.search(r"(\d{6})", os.path.basename(path))
    return int(m.group(1)) if m else None


def _discover_multi_object_est_dirs(est_dir):
    """检测 est_dir 是否为『每个螺母一个子目录』的多目标估计结果结构
    （如 foundationpose_api.py 多目标 CLI 输出的 output/poses/square_nut/、
    output/poses/square_nut_2/），是则返回 {螺母名: 子目录路径}；不是
    （目录下直接是 *.txt，或目录不存在）则返回 None，调用方应退回单目标模式。
    """
    if not est_dir or not os.path.isdir(est_dir):
        return None
    result = {}
    for entry in sorted(os.listdir(est_dir)):
        sub = os.path.join(est_dir, entry)
        if os.path.isdir(sub) and glob.glob(os.path.join(sub, "*.txt")):
            result[entry] = sub
    return result or None


def _run_batch_multi_object(model, data, est_dirs, args):
    """多目标批量对比：对 est_dirs 中每个螺母子目录分别与 --gt_dir 中的真值
    按帧号配对对比，每个目标各自汇总后，再额外汇总一份跨目标的整体统计。"""
    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "frame_*.json")))
    gt_by_idx = {_frame_index(p): p for p in gt_files}

    print(f"[多目标批量对比] 检测到 {len(est_dirs)} 个目标: {list(est_dirs.keys())}")

    per_object_summary = {}
    all_rot_errs, all_trans_errs = [], []

    for nut_name, sub_dir in est_dirs.items():
        est_files = sorted(glob.glob(os.path.join(sub_dir, "*.txt")))
        pairs = []
        for ep in est_files:
            idx = _frame_index(ep)
            if idx in gt_by_idx:
                pairs.append((ep, gt_by_idx[idx]))

        if not pairs:
            print(f"[警告] 目标 '{nut_name}': 在 {sub_dir} 与 {args.gt_dir} 中未找到可配对的帧，跳过。")
            continue

        print(f"\n{'#' * 65}\n  目标: {nut_name}  （共 {len(pairs)} 对可配对的帧）\n{'#' * 65}")

        per_frame_results = []
        rot_errs, trans_errs = [], []
        for ep, gp in pairs:
            result = compare_one(model, data, ep, gp, nut_name=nut_name, verbose=True)
            per_frame_results.append(result)
            rot_errs.append(result["rotation_error_deg"])
            trans_errs.append(result["translation_mm"]["error_norm_mm"])

        ra, ta = np.array(rot_errs), np.array(trans_errs)
        grade, tip = quality_grade(ra.mean(), ta.mean())

        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  目标 '{nut_name}' 批量统计汇总")
        print(sep)
        print(f"  {'':>10}  {'旋转误差 (\u00b0)':>14}  {'平移误差 (mm)':>16}")
        print(f"  {'均值':>10}  {ra.mean():>14.4f}  {ta.mean():>16.4f}")
        print(f"  {'最大值':>10}  {ra.max():>14.4f}  {ta.max():>16.4f}")
        print(f"  {'标准差':>10}  {ra.std():>14.4f}  {ta.std():>16.4f}")
        print(f"\n  综合质量等级: {grade}  —— {tip}")
        print(sep)

        per_object_summary[nut_name] = {
            "n_pairs": len(pairs),
            "per_frame_results": per_frame_results,
            "summary": {
                "rotation_error_deg": {"mean": float(ra.mean()), "max": float(ra.max()), "std": float(ra.std())},
                "translation_error_mm": {"mean": float(ta.mean()), "max": float(ta.max()), "std": float(ta.std())},
                "quality_grade": grade,
                "quality_tip": tip,
            },
        }
        all_rot_errs.extend(rot_errs)
        all_trans_errs.extend(trans_errs)

    if not per_object_summary:
        print("[错误] 未找到任何可配对的帧，请检查 --est_dir / --gt_dir。")
        return

    ra_all, ta_all = np.array(all_rot_errs), np.array(all_trans_errs)
    overall_grade, overall_tip = quality_grade(ra_all.mean(), ta_all.mean())

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  全部目标合计统计（{len(per_object_summary)} 个目标，共 {len(ra_all)} 组）")
    print(sep)
    print(f"  {'':>10}  {'旋转误差 (\u00b0)':>14}  {'平移误差 (mm)':>16}")
    print(f"  {'均值':>10}  {ra_all.mean():>14.4f}  {ta_all.mean():>16.4f}")
    print(f"  {'最大值':>10}  {ra_all.max():>14.4f}  {ta_all.max():>16.4f}")
    print(f"  {'标准差':>10}  {ra_all.std():>14.4f}  {ta_all.std():>16.4f}")
    print(f"\n  综合质量等级: {overall_grade}  —— {overall_tip}")
    print(sep)

    if args.save_result:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_result)), exist_ok=True)
        summary = {
            "mode": "batch_multi_object",
            "objects": list(per_object_summary.keys()),
            "per_object": per_object_summary,
            "overall_summary": {
                "n_total_pairs": int(len(ra_all)),
                "rotation_error_deg": {"mean": float(ra_all.mean()), "max": float(ra_all.max()), "std": float(ra_all.std())},
                "translation_error_mm": {"mean": float(ta_all.mean()), "max": float(ta_all.max()), "std": float(ta_all.std())},
                "quality_grade": overall_grade,
                "quality_tip": overall_tip,
            },
        }
        with open(args.save_result, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[保存] 精度指标已保存至 {args.save_result}")


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="FoundationPose 姿态估计结果 vs MuJoCo 建模真值 对比验证",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xml", default=os.path.join(here, "gen3_with_nut.xml"))
    parser.add_argument("--camera_name", default=CAMERA_NAME)

    parser.add_argument("--est_pose", default=None,
                        help="单帧估计位姿文件（如 output/poses/000000.txt）")
    parser.add_argument("--gt_pose", default=None,
                        help="单帧真值 json（如 pose_estimation_data/poses/frame_000000.json）")
    parser.add_argument("--nut_name", default=None,
                        help="仅单帧模式（--est_pose/--gt_pose）使用：指定要核验的螺母名（对应 --gt_pose "
                             "json 中 nuts_pose_world_4x4 的 key，如 square_nut_2）。省略则优先用旧版"
                             "单螺母字段 nut_pose_world_4x4（向后兼容）。")

    parser.add_argument("--est_dir", default=os.path.join(here, "output", "poses"),
                        help="批量模式：估计位姿目录。支持两种结构：单目标 <est_dir>/*.txt（文件名含 6 位"
                             "帧号）；多目标 <est_dir>/<螺母名>/*.txt（每个螺母一个子目录，如"
                             "foundationpose_api.py 多目标 CLI 输出的 poses/square_nut/、poses/square_nut_2/），"
                             "检测到该结构会自动进入多目标批量对比模式。")
    parser.add_argument("--gt_dir", default=os.path.join(here, "pose_estimation_data", "poses"),
                        help="批量模式：真值目录（frame_*.json，单/多目标场景都用同一个目录）")
    parser.add_argument("--save_result", default=None,
                        help="将精度指标结果保存为 json 文件的路径（如 output/pose_verify_result.json）。"
                             "省略则不保存。")
    args = parser.parse_args()

    print(f"[加载模型] {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    if args.est_pose and args.gt_pose:
        # ── 单帧模式 ──
        result = compare_one(model, data, args.est_pose, args.gt_pose, nut_name=args.nut_name)
        if args.save_result:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_result)), exist_ok=True)
            with open(args.save_result, "w", encoding="utf-8") as f:
                json.dump({"mode": "single", "result": result}, f, indent=2, ensure_ascii=False)
            print(f"\n[保存] 精度指标已保存至 {args.save_result}")
        return

    # ── 多目标批量模式检测：--est_dir 下是否为『每个螺母一个子目录』的结构 ──
    multi_est_dirs = _discover_multi_object_est_dirs(args.est_dir)
    if multi_est_dirs:
        _run_batch_multi_object(model, data, multi_est_dirs, args)
        return

    # ── 批量模式（单目标）：按帧号自动配对 ──
    est_files = sorted(glob.glob(os.path.join(args.est_dir, "*.txt")))
    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "frame_*.json")))

    gt_by_idx = {_frame_index(p): p for p in gt_files}
    pairs = []
    for ep in est_files:
        idx = _frame_index(ep)
        if idx in gt_by_idx:
            pairs.append((ep, gt_by_idx[idx]))

    if not pairs:
        print(f"[错误] 在 {args.est_dir} 与 {args.gt_dir} 中未找到可配对的帧文件。\n"
              f"请通过 --est_pose / --gt_pose 指定单帧文件，或检查目录及文件命名。")
        return

    print(f"[批量对比] 共找到 {len(pairs)} 对可配对的帧")

    per_frame_results = []
    rot_errs, trans_errs = [], []
    for ep, gp in pairs:
        result = compare_one(model, data, ep, gp, verbose=True)
        per_frame_results.append(result)
        rot_errs.append(result["rotation_error_deg"])
        trans_errs.append(result["translation_mm"]["error_norm_mm"])

    ra, ta = np.array(rot_errs), np.array(trans_errs)
    sep = "=" * 65
    print(f"\n{sep}")
    print("  批量统计汇总")
    print(sep)
    print(f"  {'':>10}  {'旋转误差 (°)':>14}  {'平移误差 (mm)':>16}")
    print(f"  {'均值':>10}  {ra.mean():>14.4f}  {ta.mean():>16.4f}")
    print(f"  {'最大值':>10}  {ra.max():>14.4f}  {ta.max():>16.4f}")
    print(f"  {'标准差':>10}  {ra.std():>14.4f}  {ta.std():>16.4f}")
    grade, tip = quality_grade(ra.mean(), ta.mean())
    print(f"\n  综合质量等级: {grade}  —— {tip}")
    print(sep)

    if args.save_result:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_result)), exist_ok=True)
        summary = {
            "mode": "batch",
            "n_pairs": len(pairs),
            "per_frame_results": per_frame_results,
            "summary": {
                "rotation_error_deg": {
                    "mean": float(ra.mean()),
                    "max": float(ra.max()),
                    "std": float(ra.std()),
                },
                "translation_error_mm": {
                    "mean": float(ta.mean()),
                    "max": float(ta.max()),
                    "std": float(ta.std()),
                },
                "quality_grade": grade,
                "quality_tip": tip,
            },
        }
        with open(args.save_result, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[保存] 精度指标已保存至 {args.save_result}")


if __name__ == "__main__":
    main()
