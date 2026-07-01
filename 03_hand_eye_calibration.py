#!/usr/bin/env python3
"""
03_hand_eye_calibration.py
---------------------------
基于棋盘格标定板，利用已保存的图像和位姿执行手眼标定，
同时给出重投影误差和标定一致性误差两项精度评估。

支持两种模式（--mode）：
  eye_in_hand  : 眼在手上（相机安装在机械臂末端）
                 求解 T_camera→gripper
  eye_to_hand  : 眼在手外（相机固定，板安装在机械臂上）
                 求解 T_camera→base

支持 5 种求解算法（--method）：
  tsai / park / horaud / andreff / daniilidis

精度评估：
  1. 重投影误差   : 对每张图像用 solvePnP 计算角点重投影 RMSE（像素）
  2. 标定一致性误差: 对所有位姿对验证 AX=XB 方程的残差
                    - 旋转误差（度）
                    - 平移误差（mm）
  3. 综合质量等级 : EXCELLENT / GOOD / ACCEPTABLE / POOR

用法示例：
  # 眼在手上，默认参数（9×6 棋盘格，25mm 格子）
  python 03_hand_eye_calibration.py --mode eye_in_hand

  # 眼在手外，自定义棋盘格
  python 03_hand_eye_calibration.py --mode eye_to_hand \\
      --board_width 11 --board_height 8 --square_size 0.020

  # 可视化检测过程
  python 03_hand_eye_calibration.py --mode eye_in_hand --visualize

输出：
  <data_dir>/hand_eye_result_<mode>.json
  <data_dir>/hand_eye_T_cam2end_<mode>.npy
"""

import argparse
import json
import os
import sys
from typing import Optional

import cv2
import numpy as np

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    print("[ERROR] 未找到 scipy，请先安装: pip install scipy")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────────────────────

def load_intrinsics(intrinsics_path: str):
    """从 JSON 文件加载相机内参。"""
    with open(intrinsics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
    return camera_matrix, dist_coeffs


# ──────────────────────────────────────────────────────────────────────────────
# 棋盘格检测
# ──────────────────────────────────────────────────────────────────────────────

def detect_chessboard(
    image: np.ndarray,
    board_size: tuple[int, int],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    square_size: float,
) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    检测棋盘格角点并求解相机→棋盘格的位姿。

    参数
    ----
    board_size  : (cols, rows) 内角点数（宽×高）
    square_size : 格子边长（米）

    返回
    ----
    success, corners (N,1,2), rvec (3,1), tvec (3,1)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             + cv2.CALIB_CB_FAST_CHECK
             + cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret, corners = cv2.findChessboardCorners(gray, board_size, flags)

    if not ret:
        return False, None, None, None

    # 亚像素精化
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)

    # 构建棋盘格 3D 点（z=0，单位：米）
    objp = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    ok, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return False, None, None, None

    return True, corners, rvec, tvec


# ──────────────────────────────────────────────────────────────────────────────
# 精度评估
# ──────────────────────────────────────────────────────────────────────────────

def reprojection_error(
    detections: list[dict],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    board_size: tuple[int, int],
    square_size: float,
) -> tuple[float, float, list[float]]:
    """
    计算每张图像的角点重投影 RMSE（像素）。

    返回 (mean, std, per_image_errors)
    """
    objp = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    per_img = []
    for det in detections:
        projected, _ = cv2.projectPoints(
            objp, det["rvec"], det["tvec"], camera_matrix, dist_coeffs
        )
        err = np.sqrt(np.mean((det["corners"] - projected) ** 2))
        per_img.append(float(err))

    return float(np.mean(per_img)), float(np.std(per_img)), per_img


def consistency_error(
    R_robot_list: list[np.ndarray],
    t_robot_list: list[np.ndarray],
    R_target2cam_list: list[np.ndarray],
    t_target2cam_list: list[np.ndarray],
    R_result: np.ndarray,
    t_result: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    计算手眼标定一致性误差（验证 AX = XB）。

    A 为机器人相对运动（由 R_robot/t_robot 构造）。
    B 为目标在相机系中的相对运动。
    X 为标定结果。

    返回
    ----
    (rot_mean_deg, rot_std_deg, trans_mean_mm, trans_std_mm)
    """
    n = len(R_robot_list)

    # 构造 X
    X = np.eye(4, dtype=np.float64)
    X[:3, :3] = R_result
    X[:3, 3] = t_result.flatten()

    def build_T(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T

    rot_errors = []
    trans_errors = []

    for i in range(n):
        for j in range(i + 1, n):
            Ti = build_T(R_robot_list[i], t_robot_list[i])
            Tj = build_T(R_robot_list[j], t_robot_list[j])
            A_ij = np.linalg.inv(Tj) @ Ti  # 相对机器人运动

            Bi = build_T(R_target2cam_list[i], t_target2cam_list[i])
            Bj = build_T(R_target2cam_list[j], t_target2cam_list[j])
            B_ij = Bi @ np.linalg.inv(Bj)  # 相对目标运动

            AX = A_ij @ X
            XB = X @ B_ij

            # 旋转误差（弧度 → 度）
            R_err = AX[:3, :3] @ XB[:3, :3].T
            cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
            rot_errors.append(float(np.degrees(np.arccos(cos_angle))))

            # 平移误差（米 → mm）
            trans_errors.append(float(np.linalg.norm(AX[:3, 3] - XB[:3, 3]) * 1000.0))

    return (float(np.mean(rot_errors)), float(np.std(rot_errors)),
            float(np.mean(trans_errors)), float(np.std(trans_errors)))


def board_stability_error(
    R_gripper2base_list: list[np.ndarray],
    t_gripper2base_list: list[np.ndarray],
    R_target2cam_list: list[np.ndarray],
    t_target2cam_list: list[np.ndarray],
    R_result: np.ndarray,
    t_result: np.ndarray,
    mode: str,
) -> tuple[float, float]:
    """
    计算标定板在参考坐标系中的位置稳定性（额外一致性指标）。

    eye_in_hand  : T_board_base = T_g2b × T_cam2gripper × T_target2cam
                   → 理想情况下所有帧的 T_board_base 相同（板固定）
    eye_to_hand  : T_board_cam  = T_cam2base^{-1} × T_g2b × T_board2g
                   → 本函数仅给出 eye_in_hand 下的稳定性

    返回 (mean_trans_std_mm, mean_rot_std_deg)
    """
    if mode != "eye_in_hand":
        return float("nan"), float("nan")

    X = np.eye(4, dtype=np.float64)
    X[:3, :3] = R_result
    X[:3, 3] = t_result.flatten()

    positions = []
    for R_g2b, t_g2b, R_t2c, t_t2c in zip(
        R_gripper2base_list, t_gripper2base_list,
        R_target2cam_list, t_target2cam_list
    ):
        T_g2b = np.eye(4)
        T_g2b[:3, :3] = R_g2b
        T_g2b[:3, 3] = t_g2b.flatten()

        T_t2c = np.eye(4)
        T_t2c[:3, :3] = R_t2c
        T_t2c[:3, 3] = t_t2c.flatten()

        T_board_base = T_g2b @ X @ T_t2c
        positions.append(T_board_base[:3, 3] * 1000.0)  # mm

    positions = np.array(positions)
    std_mm = float(np.mean(np.std(positions, axis=0)))
    return std_mm, float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# 检测角点 vs 重投影可视化
# ──────────────────────────────────────────────────────────────────────────────

def save_verification_images(
    detections: list[dict],
    images_dir: str,
    output_dir: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    board_size: tuple[int, int],
    square_size: float,
) -> None:
    """
    对每张有效图像绘制：
      绿色实心圆  —— 检测角点
      红色实心圆  —— 重投影角点（基于 solvePnP 结果）
      黄色细线    —— 连接对应点对，直观展示误差大小与方向

    保存路径： <output_dir>/verify_XXX.png
    """
    os.makedirs(output_dir, exist_ok=True)

    objp = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    for det in detections:
        img_path = os.path.join(images_dir, det["image"])
        vis = cv2.imread(img_path)
        if vis is None:
            continue

        projected, _ = cv2.projectPoints(
            objp, det["rvec"], det["tvec"], camera_matrix, dist_coeffs
        )

        per_pt_err = np.sqrt(
            np.sum((det["corners"].reshape(-1, 2)
                    - projected.reshape(-1, 2)) ** 2, axis=1)
        )
        rmse = float(np.sqrt(np.mean(per_pt_err ** 2)))

        # 逐点绘制
        for pt_det, pt_proj, err_pt in zip(
            det["corners"].reshape(-1, 2),
            projected.reshape(-1, 2),
            per_pt_err,
        ):
            pd = tuple(pt_det.astype(int))
            pp = tuple(pt_proj.astype(int))
            cv2.line(vis, pd, pp, (0, 220, 220), 1, cv2.LINE_AA)   # 黄色连线
            cv2.circle(vis, pd, 5, (0, 230, 0), -1, cv2.LINE_AA)   # 绿色：检测点
            cv2.circle(vis, pp, 4, (0, 0, 230), -1, cv2.LINE_AA)   # 红色：重投影点

        # 添加文字注解
        h = vis.shape[0]
        overlay_lines = [
            (f"标定图像: {det['image']}",  (10, 32),  (255, 255, 255), 0.75),
            (f"RMSE: {rmse:.4f} px",          (10, 62),  (0, 230, 230),   0.85),
            ("● 绿色: 检测角点",             (10, h - 56), (0, 230, 0),   0.70),
            ("● 红色: 重投影角点",           (10, h - 28), (0, 0, 230),   0.70),
        ]
        for text, pos, color, scale in overlay_lines:
            cv2.putText(vis, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0, 0, 0), 4, cv2.LINE_AA)   # 黑色描边
            cv2.putText(vis, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 2, cv2.LINE_AA)

        out_name = det["image"].replace("calib_image_", "verify_")
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, vis)
        print(f"  [验证图] {out_path}  RMSE={rmse:.4f} px")


# ──────────────────────────────────────────────────────────────────────────────

def quality_grade(reproj_px: float, rot_deg: float, trans_mm: float) -> str:
    if reproj_px < 1.0 and rot_deg < 1.0 and trans_mm < 5.0:
        return "EXCELLENT"
    if reproj_px < 2.0 and rot_deg < 2.0 and trans_mm < 10.0:
        return "GOOD"
    if reproj_px < 5.0 and rot_deg < 5.0 and trans_mm < 20.0:
        return "ACCEPTABLE"
    return "POOR"


QUALITY_TIPS = {
    "EXCELLENT": "标定质量极佳，可直接用于生产。",
    "GOOD": "标定质量良好，满足大多数应用需求。",
    "ACCEPTABLE": "标定质量尚可，建议增加标定位姿或重拍部分模糊图像后重新标定。",
    "POOR": (
        "标定质量较差，建议检查以下问题：\n"
        "  · 棋盘格是否平整刚性？\n"
        "  · 图像是否清晰（移动模糊、对焦不准）？\n"
        "  · 机械臂位姿是否足够多样（建议 15～20 组，覆盖不同方向）？\n"
        "  · 示教器位姿数据与图像是否一一对应？\n"
        "  · 旋转表示类型（--rotation_type）是否与机器人一致？"
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="手眼标定工具（支持眼在手上 / 眼在手外）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["eye_in_hand", "eye_to_hand"], default="eye_in_hand",
        help="标定模式: eye_in_hand（相机在末端）/ eye_to_hand（相机固定）",
    )
    parser.add_argument("--data_dir", default="calibration_data",
                        help="标定数据根目录（包含 images/ 和 poses/ 子目录）")
    parser.add_argument("--board_width", type=int, default=9,
                        help="棋盘格每行内角点数（水平方向）")
    parser.add_argument("--board_height", type=int, default=6,
                        help="棋盘格每列内角点数（垂直方向）")
    parser.add_argument("--square_size", type=float, default=0.025,
                        help="棋盘格格子边长（米，默认 0.025m = 25mm）")
    parser.add_argument(
        "--method",
        choices=["tsai", "park", "horaud", "andreff", "daniilidis"],
        default="tsai",
        help="手眼标定求解算法",
    )
    parser.add_argument("--visualize", action="store_true",
                        help="可视化每张图像的角点检测结果")
    args = parser.parse_args()

    method_map = {
        "tsai":       cv2.CALIB_HAND_EYE_TSAI,
        "park":       cv2.CALIB_HAND_EYE_PARK,
        "horaud":     cv2.CALIB_HAND_EYE_HORAUD,
        "andreff":    cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    board_size = (args.board_width, args.board_height)

    # ── 加载相机内参 ──────────────────────────────────────────────────────────
    intrinsics_path = os.path.join(args.data_dir, "camera_intrinsics.json")
    if not os.path.isfile(intrinsics_path):
        print(f"[ERROR] 找不到相机内参文件: {intrinsics_path}")
        print("请先运行 01_capture_calibration_images.py 获取内参。")
        sys.exit(1)
    camera_matrix, dist_coeffs = load_intrinsics(intrinsics_path)
    print(f"[OK] 加载相机内参: {intrinsics_path}")

    # ── 收集图像-位姿对 ───────────────────────────────────────────────────────
    images_dir = os.path.join(args.data_dir, "images")
    poses_dir = os.path.join(args.data_dir, "poses")

    image_files = sorted(
        f for f in os.listdir(images_dir)
        if f.startswith("calib_image_") and f.endswith(".png")
    )

    print(f"\n发现图像: {len(image_files)} 张")
    print(f"标定模式: {args.mode}")
    print(f"棋盘格  : {board_size[0]}×{board_size[1]} 内角点，"
          f"{args.square_size * 1000:.1f}mm 格子")
    print(f"求解算法: {args.method}")
    print()

    R_gripper2base_list, t_gripper2base_list = [], []
    R_target2cam_list,  t_target2cam_list   = [], []
    detections: list[dict] = []
    valid_indices: list[int] = []
    skip_count = 0

    for img_file in image_files:
        idx = int(img_file[len("calib_image_"):-len(".png")])
        pose_file = f"calib_pose_{idx:03d}.json"
        pose_path = os.path.join(poses_dir, pose_file)

        # 检查对应位姿是否存在
        if not os.path.isfile(pose_path):
            print(f"  [跳过] {img_file}  ← 找不到对应位姿 {pose_file}")
            skip_count += 1
            continue

        # 加载图像
        img_path = os.path.join(images_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"  [跳过] {img_file}  ← 图像读取失败")
            skip_count += 1
            continue

        # 棋盘格检测
        success, corners, rvec, tvec = detect_chessboard(
            image, board_size, camera_matrix, dist_coeffs, args.square_size
        )
        if not success:
            print(f"  [跳过] {img_file}  ← 棋盘格未检测到（检查角点数或图像质量）")
            skip_count += 1
            continue

        # 加载机器人位姿（gripper → base）—— 从 JSON 读取
        with open(pose_path, "r", encoding="utf-8") as pf:
            pose_data = json.load(pf)
        T_g2b = np.array(pose_data["matrix_4x4"], dtype=np.float64)

        R_g2b = T_g2b[:3, :3]
        t_g2b = T_g2b[:3, 3].reshape(3, 1)
        R_t2c, _ = cv2.Rodrigues(rvec)

        R_gripper2base_list.append(R_g2b)
        t_gripper2base_list.append(t_g2b)
        R_target2cam_list.append(R_t2c)
        t_target2cam_list.append(tvec)
        detections.append({
            "image": img_file,
            "corners": corners,
            "rvec": rvec,
            "tvec": tvec,
        })
        valid_indices.append(idx)
        print(f"  [OK]   {img_file} + {pose_file}")

        # 可视化
        if args.visualize:
            vis = image.copy()
            cv2.drawChessboardCorners(vis, board_size, corners, True)
            label = f"{img_file}  idx={idx}"
            cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("棋盘格检测", vis)
            key = cv2.waitKey(800) & 0xFF
            if key in (ord("q"), 27):
                args.visualize = False

    if args.visualize:
        cv2.destroyAllWindows()

    n_valid = len(valid_indices)
    print(f"\n有效数据对: {n_valid}  （跳过: {skip_count}）")

    if n_valid < 3:
        print("[ERROR] 有效数据对不足 3 组，无法进行手眼标定！")
        sys.exit(1)

    # ── 为 eye_to_hand 准备 base2gripper ────────────────────────────────────
    if args.mode == "eye_to_hand":
        R_robot_list, t_robot_list = [], []
        for R, t in zip(R_gripper2base_list, t_gripper2base_list):
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()
            T_inv = np.linalg.inv(T)
            R_robot_list.append(T_inv[:3, :3])
            t_robot_list.append(T_inv[:3, 3].reshape(3, 1))
    else:
        R_robot_list = R_gripper2base_list
        t_robot_list = t_gripper2base_list

    # ── 执行手眼标定 ──────────────────────────────────────────────────────────
    print(f"\n正在执行手眼标定（{args.mode}，方法: {args.method}）...")
    R_result, t_result = cv2.calibrateHandEye(
        R_robot_list,      t_robot_list,
        R_target2cam_list, t_target2cam_list,
        method=method_map[args.method],
    )

    T_result = np.eye(4, dtype=np.float64)
    T_result[:3, :3] = R_result
    T_result[:3, 3] = t_result.flatten()

    euler_deg = Rotation.from_matrix(R_result).as_euler("XYZ", degrees=True)
    quat_xyzw = Rotation.from_matrix(R_result).as_quat().tolist()

    # ── 精度评估 ──────────────────────────────────────────────────────────────
    reproj_mean, reproj_std, reproj_per_img = reprojection_error(
        detections, camera_matrix, dist_coeffs, board_size, args.square_size
    )

    rot_mean, rot_std, trans_mean, trans_std = consistency_error(
        R_robot_list,      t_robot_list,
        R_target2cam_list, t_target2cam_list,
        R_result, t_result,
    )

    stab_trans_mm, _ = board_stability_error(
        R_gripper2base_list, t_gripper2base_list,
        R_target2cam_list, t_target2cam_list,
        R_result, t_result, args.mode,
    )

    grade = quality_grade(reproj_mean, rot_mean, trans_mean)

    # ── 保存验证图像 ──────────────────────────────────────────────────────────
    verify_dir = os.path.join(args.data_dir, "verification")
    print(f"\n正在生成验证图像...")
    save_verification_images(
        detections, images_dir, verify_dir,
        camera_matrix, dist_coeffs, board_size, args.square_size,
    )

    sep = "=" * 60
    print(f"\n{sep}")
    if args.mode == "eye_in_hand":
        print("手眼标定结果  T_camera → gripper（眼在手上）")
    else:
        print("手眼标定结果  T_camera → base（眼在手外）")
    print(sep)
    print("4×4 变换矩阵：")
    for row in T_result:
        print("  " + "  ".join(f"{v:10.6f}" for v in row))
    print(f"\n平移（mm）  : [{t_result[0,0]*1000:.3f},  {t_result[1,0]*1000:.3f},  {t_result[2,0]*1000:.3f}]")
    print(f"旋转（XYZ 欧拉角，度）: [{euler_deg[0]:.4f},  {euler_deg[1]:.4f},  {euler_deg[2]:.4f}]")
    print(f"旋转（四元数 x,y,z,w）: {[f'{v:.6f}' for v in quat_xyzw]}")

    print(f"\n{sep}")
    print("精度评估")
    print(sep)
    print(f"重投影误差（角点）   : {reproj_mean:.4f} ± {reproj_std:.4f} 像素")
    for det, e in zip(detections, reproj_per_img):
        print(f"    {det['image']}:  {e:.4f} px")
    print(f"\n标定一致性误差（AX=XB）:")
    print(f"    旋转误差  : {rot_mean:.4f} ± {rot_std:.4f} 度")
    print(f"    平移误差  : {trans_mean:.4f} ± {trans_std:.4f} mm")
    if not np.isnan(stab_trans_mm):
        print(f"\n标定板位置稳定性（眼在手上）:")
        print(f"    位置标准差: {stab_trans_mm:.4f} mm")

    print(f"\n综合质量等级: {grade}")
    print(f"  {QUALITY_TIPS[grade]}")
    print(sep)

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    result = {
        "mode": args.mode,
        "method": args.method,
        "n_valid_pairs": n_valid,
        "valid_indices": valid_indices,
        "board_size_wh": board_size,
        "square_size_m": args.square_size,
        "T_cam2end_4x4": T_result.tolist(),
        "R_cam2end_3x3": R_result.tolist(),
        "t_cam2end_m":   t_result.flatten().tolist(),
        "t_cam2end_mm":  (t_result.flatten() * 1000).tolist(),
        "euler_XYZ_degrees": euler_deg.tolist(),
        "quaternion_xyzw": quat_xyzw,
        "accuracy": {
            "reprojection_error_mean_px":           reproj_mean,
            "reprojection_error_std_px":            reproj_std,
            "reprojection_error_per_image_px":      dict(
                zip([d["image"] for d in detections], reproj_per_img)
            ),
            "consistency_rotation_error_mean_deg":  rot_mean,
            "consistency_rotation_error_std_deg":   rot_std,
            "consistency_translation_error_mean_mm": trans_mean,
            "consistency_translation_error_std_mm":  trans_std,
            "board_position_stability_mm": (
                stab_trans_mm if not np.isnan(stab_trans_mm) else None
            ),
            "quality_grade": grade,
        },
    }

    json_path = os.path.join(args.data_dir, f"hand_eye_result_{args.mode}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    npy_path = os.path.join(args.data_dir, f"hand_eye_T_cam2end_{args.mode}.npy")
    np.save(npy_path, T_result)

    print(f"\n结果已保存:")
    print(f"  标定结果 JSON : {json_path}")
    print(f"  变换矩阵 .npy : {npy_path}")
    print(f"  验证图像目录  : {verify_dir}")
    print(sep)


if __name__ == "__main__":
    main()
