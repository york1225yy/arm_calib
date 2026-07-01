#!/usr/bin/env python3
"""
02_convert_tcp_poses.py
------------------------
将示教器获取的 TCP 坐标（xyz mm, rx ry rz rad）批量转换为
4×4 齐次变换矩阵，保存为与图像一一对应的 .npy 文件。

输入文件格式（tcp_poses.txt，每行一个位姿，# 开头为注释）：
  x(mm)  y(mm)  z(mm)  rx(rad)  ry(rad)  rz(rad)
  例：
    300.12  -45.67  500.00  0.0   1.5708  0.0
    #  对应图像 calib_image_001.png

旋转表示方式（--rotation_type）：
  euler_zyx  : ZYX 外旋欧拉角（绕固定轴 X→Y→Z，即 scipy 的 'xyz' 小写）
               适用：KUKA / Fanuc / 大多数工业机器人示教器
  euler_xyz  : XYZ 外旋欧拉角（绕固定轴 Z→Y→X，即 scipy 的 'zyx' 小写）
               适用：部分机器人（绕自身坐标轴 X,Y,Z 依次旋转）
  rotation_vector : 旋转向量（轴角，‖[rx,ry,rz]‖ 为旋转角）
               适用：UR 系列机器人

输出：
  <output_dir>/poses/calib_pose_XXX.json  —— 4×4 矩阵及可读字段（平移单位: 米）
  <output_dir>/poses_summary.json         —— 所有位姿的汇总信息

用法示例：
  python 02_convert_tcp_poses.py --input tcp_poses.txt
  python 02_convert_tcp_poses.py --input tcp_poses.txt --rotation_type rotation_vector
  python 02_convert_tcp_poses.py --input tcp_poses.txt --rotation_type euler_zyx --start_index 1
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    print("[ERROR] 未找到 scipy，请先安装: pip install scipy")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# 核心转换
# ──────────────────────────────────────────────────────────────────────────────

ROTATION_TYPES = ("euler_zyx", "euler_xyz", "rotation_vector")


def tcp_to_matrix(x_mm: float, y_mm: float, z_mm: float,
                  rx: float, ry: float, rz: float,
                  rotation_type: str) -> np.ndarray:
    """
    将 TCP 位姿转换为 4×4 齐次变换矩阵（平移单位转为米）。

    参数
    ----
    x_mm, y_mm, z_mm : 平移量（毫米）
    rx, ry, rz       : 旋转量（弧度）
    rotation_type    : 旋转表示方式

    返回
    ----
    T : shape (4, 4) float64，gripper → base 的变换矩阵
    """
    t = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0  # mm → m

    if rotation_type == "euler_zyx":
        # 外旋 ZYX ↔ 内旋 XYZ: Rotation.from_euler('XYZ', [rx,ry,rz])
        R = Rotation.from_euler("XYZ", [rx, ry, rz]).as_matrix()
    elif rotation_type == "euler_xyz":
        # 外旋 XYZ ↔ 内旋 ZYX: Rotation.from_euler('ZYX', [rz,ry,rx])
        R = Rotation.from_euler("ZYX", [rz, ry, rx]).as_matrix()
    elif rotation_type == "rotation_vector":
        # 旋转向量（轴角）
        R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
    else:
        raise ValueError(f"未知旋转类型: {rotation_type}，支持: {ROTATION_TYPES}")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def matrix_summary(T: np.ndarray) -> dict:
    """提取变换矩阵的可读摘要（欧拉角 + 平移）。"""
    R_obj = Rotation.from_matrix(T[:3, :3])
    euler_xyz = R_obj.as_euler("XYZ", degrees=True).tolist()
    quat = R_obj.as_quat().tolist()  # [x, y, z, w]
    return {
        "translation_mm": (T[:3, 3] * 1000).tolist(),
        "rotation_euler_XYZ_deg": euler_xyz,
        "rotation_quaternion_xyzw": quat,
        "matrix_4x4": T.tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TCP 位姿 → 4×4 变换矩阵 (.json) 转换工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="tcp_poses.txt",
                        help="TCP 位姿输入文件（每行: x y z rx ry rz）")
    parser.add_argument("--output_dir", default="calibration_data",
                        help="输出根目录（与图像目录保持一致）")
    parser.add_argument(
        "--rotation_type", default="euler_zyx",
        choices=ROTATION_TYPES,
        help=(
            "旋转表示类型:\n"
            "  euler_zyx        = ZYX 外旋欧拉角（KUKA/Fanuc 等）\n"
            "  euler_xyz        = XYZ 外旋欧拉角\n"
            "  rotation_vector  = 旋转向量/轴角（UR 机器人）"
        ),
    )
    parser.add_argument("--start_index", type=int, default=1,
                        help="起始编号，需与对应图像编号一致")
    args = parser.parse_args()

    # ── 检查输入文件 ──────────────────────────────────────────────────────────
    if not os.path.isfile(args.input):
        print(f"[ERROR] 输入文件不存在: {args.input}")
        print("请创建 tcp_poses.txt，每行格式: x(mm) y(mm) z(mm) rx(rad) ry(rad) rz(rad)")
        sys.exit(1)

    # ── 读取 TCP 位姿 ─────────────────────────────────────────────────────────
    poses_raw: list[tuple] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 6:
                print(f"[警告] 第 {lineno} 行格式不正确（期望 6 个数值），已跳过: {line}")
                continue
            try:
                values = tuple(float(v) for v in parts)
            except ValueError as e:
                print(f"[警告] 第 {lineno} 行解析失败: {e}，已跳过")
                continue
            poses_raw.append(values)

    if not poses_raw:
        print("[ERROR] 未读取到有效位姿，请检查输入文件格式")
        sys.exit(1)

    print("=" * 60)
    print(f"读取到 {len(poses_raw)} 个 TCP 位姿")
    print(f"旋转类型 : {args.rotation_type}")
    print(f"起始编号 : {args.start_index:03d}")
    print("=" * 60)

    # ── 创建输出目录 ──────────────────────────────────────────────────────────
    poses_dir = os.path.join(args.output_dir, "poses")
    os.makedirs(poses_dir, exist_ok=True)

    # ── 逐个转换并保存 ────────────────────────────────────────────────────────
    summary_entries = []
    for i, (x, y, z, rx, ry, rz) in enumerate(poses_raw):
        idx = args.start_index + i
        T = tcp_to_matrix(x, y, z, rx, ry, rz, args.rotation_type)

        # 保存 .json（人类可读，VSCode 可直接查看）
        filename = f"calib_pose_{idx:03d}.json"
        filepath = os.path.join(poses_dir, filename)
        pose_json = {
            "index": idx,
            "corresponding_image": f"calib_image_{idx:03d}.png",
            "tcp_input": {
                "x_mm": x, "y_mm": y, "z_mm": z,
                "rx_rad": rx, "ry_rad": ry, "rz_rad": rz,
            },
            **matrix_summary(T),
        }
        with open(filepath, "w", encoding="utf-8") as fj:
            json.dump(pose_json, fj, indent=4, ensure_ascii=False)

        entry = pose_json.copy()
        summary_entries.append(entry)

        euler_str = "[{:.3f}, {:.3f}, {:.3f}] deg".format(
            *entry["rotation_euler_XYZ_deg"])
        trans_str = "[{:.2f}, {:.2f}, {:.2f}] mm".format(
            *entry["translation_mm"])
        print(f"  [{idx:03d}] 平移={trans_str}  旋转={euler_str}  → {filename}")

    # ── 保存汇总 JSON ─────────────────────────────────────────────────────────
    summary = {
        "total_poses": len(poses_raw),
        "rotation_type": args.rotation_type,
        "start_index": args.start_index,
        "units": {
            "translation_stored": "meters",
            "translation_input": "mm",
            "rotation_input": "rad",
        },
        "poses": summary_entries,
    }
    summary_path = os.path.join(args.output_dir, "poses_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("=" * 60)
    print(f"转换完成: {len(poses_raw)} 个位姿")
    print(f"位姿目录 : {poses_dir}")
    print(f"汇总文件 : {summary_path}")
    print("=" * 60)

    # ── 检查与图像的对应关系 ──────────────────────────────────────────────────
    images_dir = os.path.join(args.output_dir, "images")
    if os.path.isdir(images_dir):
        imgs = {
            int(f[len("calib_image_"):-len(".png")])
            for f in os.listdir(images_dir)
            if f.startswith("calib_image_") and f.endswith(".png")
        }
        saved_poses = {
            int(f[len("calib_pose_"):-len(".json")])
            for f in os.listdir(poses_dir)
            if f.startswith("calib_pose_") and f.endswith(".json")
        }
        pose_indices = set(range(args.start_index, args.start_index + len(poses_raw)))
        missing_imgs = pose_indices - imgs
        missing_poses = imgs - pose_indices
        if missing_imgs:
            print(f"[警告] 以下位姿缺少对应图像: {sorted(missing_imgs)}")
        if missing_poses:
            print(f"[警告] 以下图像缺少对应位姿: {sorted(missing_poses)}")
        if not missing_imgs and not missing_poses:
            print(f"[OK] 图像与位姿完全对应（共 {len(pose_indices)} 对）")


if __name__ == "__main__":
    main()
