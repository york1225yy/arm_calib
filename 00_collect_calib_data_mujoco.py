#!/usr/bin/env python3
"""
00_collect_calib_data_mujoco.py
--------------------------------
基于 MuJoCo 仿真环境自动采集手眼标定数据：
  - 模型文件：gen3_with_camera.xml（已内嵌 9×6 棋盘格标定板）
  - 相机：d435i_rgb_camera（fovy=42.5, 640×480）
  - 采集模式：遍历预设的多组关节位姿，稳定后保存图像与位姿

输出目录结构（与 03_hand_eye_calibration.py 完全兼容）：
  <data_dir>/
    camera_intrinsics.json          ← 从仿真相机参数推算
    images/
      calib_image_000.png
      calib_image_001.png
      ...
    poses/
      calib_pose_000.json           ← gripper→base 4×4 变换矩阵
      calib_pose_001.json
      ...

用法：
  python 00_collect_calib_data_mujoco.py
  python 00_collect_calib_data_mujoco.py --data_dir my_data --settle_steps 500
  python 00_collect_calib_data_mujoco.py --preview   # 交互预览，不自动保存

关于坐标系（eye_in_hand 模式）：
  T_gripper2base = bracelet_link 在 world 系中的位姿（去掉 base_link 偏移后即 base 系）
  由于 base_link.pos = (-0.5, 0, 0.42)，
  T_gripper2base = inv(T_base_world) @ T_bracelet_world
"""

import argparse
import json
import math
import os

# ── 离屏渲染：无显示器环境使用 OSMesa 软件光栅化 ─────────────────────────────
# 必须在 import mujoco 之前设置，否则 MuJoCo 会尝试找 X11/GLFW
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco
import mujoco.renderer as mj_renderer
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────
XML_PATH = os.path.join(os.path.dirname(__file__), "gen3_with_camera.xml")
CAMERA_NAME = "d435i_rgb_camera"
CAM_WIDTH = 640
CAM_HEIGHT = 480

# 关节名称（顺序与 model 中一致）
JOINT_NAMES = ["joint_1", "joint_2", "joint_3",
               "joint_4", "joint_5", "joint_6", "joint_7"]

# 棋盘格参数（与 03_hand_eye_calibration.py 默认一致）
BOARD_WIDTH = 9    # 内角点列数
BOARD_HEIGHT = 6   # 内角点行数
SQUARE_SIZE = 0.025  # 米

# ─────────────────────────────────────────────────────────────────────────────
# 预设的标定位姿（关节角，单位：弧度）
# 策略：从 home 姿态出发，在俯仰/偏转/翻滚方向做多样化扰动，
#       保证相机正对标定板且距离适中（约 0.4～0.8 m）
# ─────────────────────────────────────────────────────────────────────────────
# fmt: off
CALIB_QPOS = [
    # home 姿态（基准）
    [0.0,    0.2618,  3.1416, -2.2689,  0.0,    0.9599,  1.5708],
    # 绕 joint_1 左转
    [-0.4,   0.2618,  3.1416, -2.2689,  0.0,    0.9599,  1.5708],
    # 绕 joint_1 右转
    [ 0.4,   0.2618,  3.1416, -2.2689,  0.0,    0.9599,  1.5708],
    # 抬高（joint_2 减小）
    [ 0.0,   0.0000,  3.1416, -2.2689,  0.0,    0.9599,  1.5708],
    # 降低（joint_2 增大）
    [ 0.0,   0.5000,  3.1416, -2.2689,  0.0,    0.9599,  1.5708],
    # 左前倾（joint_1 + joint_2）
    [-0.3,   0.1500,  3.1416, -2.2689,  0.0,    0.9599,  1.5708],
    # 右前倾
    [ 0.3,   0.1500,  3.1416, -2.2689,  0.0,    0.9599,  1.5708],
    # 腕部翻转（joint_7）
    [ 0.0,   0.2618,  3.1416, -2.2689,  0.0,    0.9599,  0.7854],
    [ 0.0,   0.2618,  3.1416, -2.2689,  0.0,    0.9599,  2.3562],
    # joint_4 变化（改变肘部）
    [ 0.0,   0.2618,  3.1416, -1.9000,  0.0,    0.9599,  1.5708],
    [ 0.0,   0.2618,  3.1416, -2.5000,  0.0,    0.9599,  1.5708],
    # joint_6 变化（改变腕俯仰）
    [ 0.0,   0.2618,  3.1416, -2.2689,  0.0,    0.7000,  1.5708],
    [ 0.0,   0.2618,  3.1416, -2.2689,  0.0,    1.2000,  1.5708],
    # joint_5 偏转
    [ 0.0,   0.2618,  3.1416, -2.2689,  0.3,    0.9599,  1.5708],
    [-0.0,   0.2618,  3.1416, -2.2689, -0.3,    0.9599,  1.5708],
    # 综合扰动
    [-0.3,   0.3500,  3.1416, -2.1000,  0.2,    0.8000,  1.3000],
    [ 0.3,   0.1500,  3.1416, -2.4000, -0.2,    1.1000,  1.8000],
    [-0.5,   0.2000,  3.1416, -2.3000,  0.0,    0.9000,  1.5708],
    [ 0.5,   0.3000,  3.1416, -2.1000,  0.0,    1.0000,  1.5708],
    [ 0.0,   0.4000,  3.1416, -1.8000,  0.0,    0.8500,  1.5708],
]
# fmt: on


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def get_camera_intrinsics(model: mujoco.MjModel, cam_name: str,
                           width: int, height: int) -> dict:
    """
    从 MuJoCo 模型中的相机 fovy 反推针孔相机内参矩阵。

    MuJoCo 相机约定：
      tan(fovy/2) = (height/2) / fy
      fx = fy * (width / height)   （假设方形像素，无畸变）
      cx = width / 2,  cy = height / 2
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1:
        raise ValueError(f"找不到相机: {cam_name}")
    fovy_rad = model.cam_fovy[cam_id] * math.pi / 180.0
    fy = (height / 2.0) / math.tan(fovy_rad / 2.0)
    fx = fy * (width / height)
    cx = width / 2.0
    cy = height / 2.0

    K = [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]]

    return {
        "camera_matrix": K,
        "dist_coeffs": [[0.0, 0.0, 0.0, 0.0, 0.0]],
        "image_width": width,
        "image_height": height,
        "fovy_deg": float(model.cam_fovy[cam_id]),
        "note": "从 MuJoCo 仿真推算，无畸变",
    }


def get_gripper2base(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """
    计算 T_gripper2base（gripper/末端坐标系 → 机械臂 base 坐标系）。

    MuJoCo 中：
      - base_link 固定在世界坐标系 pos=(-0.5, 0, 0.42)（无旋转）
      - bracelet_link 是末端 body

    T_gripper2base = inv(T_base_world) @ T_bracelet_world
    """
    # base_link 在世界系中的位置（无旋转，直接读 pos 即可）
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    bracelet_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bracelet_link")

    # 世界→base（T_base_world）
    pos_base = data.xpos[base_id].copy()     # (3,)
    rot_base = data.xmat[base_id].reshape(3, 3).copy()  # row-major

    T_base_world = np.eye(4)
    T_base_world[:3, :3] = rot_base
    T_base_world[:3, 3] = pos_base

    # 世界→bracelet（T_bracelet_world）
    pos_brac = data.xpos[bracelet_id].copy()
    rot_brac = data.xmat[bracelet_id].reshape(3, 3).copy()

    T_brac_world = np.eye(4)
    T_brac_world[:3, :3] = rot_brac
    T_brac_world[:3, 3] = pos_brac

    # T_gripper2base = inv(T_base_world) @ T_brac_world
    T_gripper2base = np.linalg.inv(T_base_world) @ T_brac_world
    return T_gripper2base


def settle_simulation(model: mujoco.MjModel, data: mujoco.MjData,
                       qpos_target: np.ndarray, n_steps: int):
    """
    将位置控制器目标设为 qpos_target，仿真 n_steps 步使其稳定。
    同时将初始 qpos 直接跳转到目标（避免大幅运动触发碰撞）。
    """
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                 for n in JOINT_NAMES]
    act_ids   = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                 for n in JOINT_NAMES]

    # 直接设置初始 qpos（加速收敛）
    for jid, q in zip(joint_ids, qpos_target):
        addr = model.jnt_qposadr[jid]
        data.qpos[addr] = q
        data.qvel[addr] = 0.0

    # 设置控制器目标
    for aid, q in zip(act_ids, qpos_target):
        data.ctrl[aid] = q

    mujoco.mj_forward(model, data)

    for _ in range(n_steps):
        mujoco.mj_step(model, data)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo 手眼标定数据采集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xml", default=XML_PATH,
                        help="MuJoCo XML 模型文件路径")
    parser.add_argument("--data_dir", default="calibration_data",
                        help="输出数据目录")
    parser.add_argument("--settle_steps", type=int, default=300,
                        help="每个位姿稳定仿真步数")
    parser.add_argument("--preview", action="store_true",
                        help="交互预览模式（按 q 退出，SPACE 保存当前帧）")
    args = parser.parse_args()

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    print(f"[加载模型] {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)  # home 姿态

    # ── 准备输出目录 ──────────────────────────────────────────────────────────
    images_dir = os.path.join(args.data_dir, "images")
    poses_dir  = os.path.join(args.data_dir, "poses")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(poses_dir,  exist_ok=True)

    # ── 保存相机内参 ──────────────────────────────────────────────────────────
    intrinsics = get_camera_intrinsics(model, CAMERA_NAME, CAM_WIDTH, CAM_HEIGHT)
    intr_path = os.path.join(args.data_dir, "camera_intrinsics.json")
    with open(intr_path, "w", encoding="utf-8") as f:
        json.dump(intrinsics, f, indent=4, ensure_ascii=False)
    print(f"[OK] 相机内参已保存: {intr_path}")
    K = np.array(intrinsics["camera_matrix"])
    print(f"     fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, "
          f"cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")

    # ── 渲染器 ────────────────────────────────────────────────────────────────
    renderer = mj_renderer.Renderer(model, height=CAM_HEIGHT, width=CAM_WIDTH)

    # ── 交互预览模式 ──────────────────────────────────────────────────────────
    if args.preview:
        _run_preview(model, data, renderer, images_dir, poses_dir, args)
        return

    # ── 自动批量采集 ──────────────────────────────────────────────────────────
    print(f"\n开始批量采集，共 {len(CALIB_QPOS)} 个位姿...")
    saved = 0
    for idx, qpos_target in enumerate(CALIB_QPOS):
        qpos_arr = np.array(qpos_target, dtype=np.float64)
        settle_simulation(model, data, qpos_arr, args.settle_steps)

        # 渲染相机视角
        renderer.update_scene(data, camera=CAMERA_NAME)
        rgb = renderer.render()          # shape (H, W, 3), RGB uint8

        # 转为 BGR 保存（OpenCV 格式）
        import cv2
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        img_name  = f"calib_image_{idx:03d}.png"
        pose_name = f"calib_pose_{idx:03d}.json"
        img_path  = os.path.join(images_dir, img_name)
        pose_path = os.path.join(poses_dir,  pose_name)

        cv2.imwrite(img_path, bgr)

        T_g2b = get_gripper2base(model, data)
        pose_data = {
            "index": idx,
            "joint_angles_rad": qpos_arr.tolist(),
            "matrix_4x4": T_g2b.tolist(),
        }
        with open(pose_path, "w", encoding="utf-8") as f:
            json.dump(pose_data, f, indent=4, ensure_ascii=False)

        print(f"  [{idx:03d}] {img_name}  +  {pose_name}  已保存")
        saved += 1

    print(f"\n[完成] 共保存 {saved} 组数据到 '{args.data_dir}/'")
    print("       可直接运行：")
    print(f"       python 03_hand_eye_calibration.py "
          f"--mode eye_in_hand --data_dir {args.data_dir}")

    renderer.close()


def _run_preview(model, data, renderer, images_dir, poses_dir, args):
    """
    交互预览：使用 OpenCV 窗口实时显示相机画面，
    支持通过键盘切换位姿和手动保存。
    """
    import cv2

    print("\n[预览模式]")
    print("  ← / →  : 上一个 / 下一个位姿")
    print("  SPACE   : 保存当前帧")
    print("  q / ESC : 退出")

    idx = 0
    n = len(CALIB_QPOS)
    saved_count = 0

    def render_current():
        qpos_arr = np.array(CALIB_QPOS[idx], dtype=np.float64)
        settle_simulation(model, data, qpos_arr, args.settle_steps)
        renderer.update_scene(data, camera=CAMERA_NAME)
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        label = f"Pose {idx}/{n-1}  |  SPACE:save  Q:quit"
        cv2.putText(bgr, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        return bgr, qpos_arr

    while True:
        frame, qpos_arr = render_current()
        cv2.imshow("MuJoCo Calib Preview", frame)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):   # q or ESC
            break
        elif key == 81 or key == ord("a"):  # left arrow
            idx = (idx - 1) % n
        elif key == 83 or key == ord("d"):  # right arrow
            idx = (idx + 1) % n
        elif key == ord(" "):
            img_name  = f"calib_image_{saved_count:03d}.png"
            pose_name = f"calib_pose_{saved_count:03d}.json"
            cv2.imwrite(os.path.join(images_dir, img_name),
                        cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            T_g2b = get_gripper2base(model, data)
            pose_data = {
                "index": saved_count,
                "joint_angles_rad": qpos_arr.tolist(),
                "matrix_4x4": T_g2b.tolist(),
            }
            with open(os.path.join(poses_dir, pose_name), "w", encoding="utf-8") as f:
                json.dump(pose_data, f, indent=4, ensure_ascii=False)
            print(f"  [保存] {img_name} + {pose_name}  (共 {saved_count+1} 组)")
            saved_count += 1

    cv2.destroyAllWindows()
    renderer.close()
    print(f"\n[退出] 共保存 {saved_count} 组数据到 '{args.data_dir}/'")


if __name__ == "__main__":
    main()
