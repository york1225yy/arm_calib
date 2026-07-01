#!/usr/bin/env python3
"""
01_capture_calibration_images.py
---------------------------------
连接 RealSense D455，获取相机内参并保存，同时提供可视化窗口。
按键说明：
  S / 空格  -- 保存当前帧（文件名 calib_image_XXX.png）
  U        -- 切换畸变校正预览（不影响保存的原始图像）
  Q / ESC  -- 退出

保存规则：
  - 原始（有畸变）图像保存至 <output_dir>/images/calib_image_XXX.png
  - 内参保存至 <output_dir>/camera_intrinsics.json
  - 编号从 001 开始，自动跳过已存在的编号，保证与后续位姿一一对应

用法示例：
  python 01_capture_calibration_images.py
  python 01_capture_calibration_images.py --output_dir calibration_data --width 1280 --height 720
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("[ERROR] 未找到 pyrealsense2，请先安装: pip install pyrealsense2")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def next_index(images_dir: str) -> int:
    """返回下一个可用的图像序号（基于已有文件自动续号）。"""
    existing = [
        f for f in os.listdir(images_dir)
        if f.startswith("calib_image_") and f.endswith(".png")
    ]
    if not existing:
        return 1
    nums = [int(f[len("calib_image_"):-len(".png")]) for f in existing]
    return max(nums) + 1


def draw_overlay(img: np.ndarray, count: int, show_undist: bool) -> np.ndarray:
    """在图像上绘制提示信息。"""
    overlay = img.copy()
    h, w = overlay.shape[:2]

    # 半透明背景条
    banner = np.zeros((60, w, 3), dtype=np.uint8)
    cv2.addWeighted(banner, 0.5, overlay[:60], 0.5, 0, overlay[:60])

    def put(text, y, color=(0, 255, 0)):
        cv2.putText(overlay, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    put(f"已保存: {count} 张  |  [S/空格] 保存  [U] 切换校正  [Q/ESC] 退出", 22)
    mode_str = "预览: 畸变校正" if show_undist else "预览: 原始"
    put(mode_str, 50, color=(0, 200, 255))
    return overlay


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RealSense D455 标定图像采集工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", default="calibration_data",
                        help="输出根目录")
    parser.add_argument("--width", type=int, default=1280,
                        help="彩色流宽度（像素）")
    parser.add_argument("--height", type=int, default=720,
                        help="彩色流高度（像素）")
    parser.add_argument("--fps", type=int, default=30,
                        help="帧率")
    args = parser.parse_args()

    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # ── 初始化 RealSense ──────────────────────────────────────────────────────
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height,
                         rs.format.bgr8, args.fps)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"[ERROR] 无法启动 RealSense 相机: {e}")
        sys.exit(1)

    # ── 获取相机内参 ──────────────────────────────────────────────────────────
    color_profile = profile.get_stream(rs.stream.color)
    intr = color_profile.as_video_stream_profile().get_intrinsics()

    camera_matrix = np.array([
        [intr.fx,      0.0, intr.ppx],
        [    0.0, intr.fy, intr.ppy],
        [    0.0,      0.0,      1.0],
    ], dtype=np.float64)
    dist_coeffs = np.array(intr.coeffs, dtype=np.float64)  # [k1,k2,p1,p2,k3]

    intrinsics_data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "distortion_model": str(intr.model),
    }
    intrinsics_path = os.path.join(args.output_dir, "camera_intrinsics.json")
    with open(intrinsics_path, "w", encoding="utf-8") as f:
        json.dump(intrinsics_data, f, indent=4, ensure_ascii=False)

    print("=" * 60)
    print("RealSense D455 相机内参")
    print("=" * 60)
    print(f"分辨率  : {intr.width} x {intr.height}")
    print(f"焦距    : fx={intr.fx:.4f}  fy={intr.fy:.4f}")
    print(f"主点    : cx={intr.ppx:.4f}  cy={intr.ppy:.4f}")
    print(f"畸变系数: {dist_coeffs.tolist()}")
    print(f"内参已保存至: {intrinsics_path}")
    print("=" * 60)
    print("按键: [S/空格] 保存  [U] 切换校正预览  [Q/ESC] 退出")
    print("=" * 60)

    # 预计算 undistort map（避免实时重算）
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix,
        (intr.width, intr.height), cv2.CV_16SC2
    )

    image_count = next_index(images_dir) - 1  # 当前已有数量
    show_undist = False

    # ── 预热：丢弃前几帧（自动曝光稳定）────────────────────────────────────
    for _ in range(30):
        pipeline.wait_for_frames()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            raw_img = np.asanyarray(color_frame.get_data())  # 原始图（保存用）

            # 预览图（可选畸变校正）
            preview = cv2.remap(raw_img, map1, map2, cv2.INTER_LINEAR) \
                if show_undist else raw_img.copy()
            preview = draw_overlay(preview, image_count, show_undist)

            cv2.imshow("RealSense D455 - 标定图像采集", preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("s"), ord("S"), ord(" ")):
                # ── 保存原始图像（未校正，供后续标定使用）──────────────────
                image_count += 1
                filename = f"calib_image_{image_count:03d}.png"
                filepath = os.path.join(images_dir, filename)
                cv2.imwrite(filepath, raw_img)
                print(f"[保存] {filepath}")

                # 短暂闪烁提示
                flash = raw_img.copy()
                cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]),
                              (0, 255, 255), 10)
                cv2.putText(flash, f"SAVED: {filename}",
                            (10, flash.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3,
                            cv2.LINE_AA)
                cv2.imshow("RealSense D455 - 标定图像采集", flash)
                cv2.waitKey(400)

            elif key in (ord("u"), ord("U")):
                show_undist = not show_undist
                mode = "畸变校正" if show_undist else "原始"
                print(f"[切换] 预览模式: {mode}")

            elif key in (ord("q"), ord("Q"), 27):  # 27 = ESC
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print("=" * 60)
    print(f"本次共保存 {image_count} 张标定图像")
    print(f"图像目录 : {images_dir}")
    print(f"内参文件 : {intrinsics_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
