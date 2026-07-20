#!/usr/bin/env python3
"""
05_collect_pose_estimation_data_mujoco.py
--------------------------------------------
基于 MuJoCo 仿真环境（gen3_with_nut.xml，机械臂 + 方形螺母）采集
6D 姿态估计（FoundationPose）所需的数据：RGB 图像 + 深度图 + 目标掩码 + 相机内参。

与 00_collect_calib_data_mujoco.py 的区别：
  - 场景换成 gen3_with_nut.xml（标定板 → 方形螺母 square_nut）
  - 每一帧除了 RGB，还额外渲染并保存：
      · 深度图（uint16 PNG，单位 mm，与 foundationpose_api.py 的
        FrameSource 读取约定一致：depth = imread(...).astype(float32)/1e3）
      · 螺母二值掩码（uint8 PNG，255=目标，0=背景，来自 MuJoCo 分割渲染）
  - 输出目录结构对齐 foundationpose_api.py 期望的输入格式，采集完成后
    可直接：
      python foundationpose_api.py --mesh <nut_mesh> \\
          --input pose_estimation_data --cam_K_file pose_estimation_data/cam_K.txt

两种模式：
  1. 交互模式（默认）：键盘手动控制各关节，看到螺母后按 SPACE 保存
  2. 复现模式（--replay）：自动复现已保存的位姿，批量重新渲染采集

键盘操作（交互模式）：
  1~7        : 选择要控制的关节
  = / +      : 所选关节角度 +step
  -          : 所选关节角度 -step
  ] / [      : step × 2 / step ÷ 2
  r          : 重置到 home 姿态
  SPACE      : 保存当前帧（RGB + 深度 + 掩码 + 末端位姿）
  q / ESC    : 退出

用法：
  python 05_collect_pose_estimation_data_mujoco.py --preview
  python 05_collect_pose_estimation_data_mujoco.py --replay --data_dir pose_estimation_data

输出目录结构：
  <data_dir>/
    cam_K.txt              相机内参 3x3 矩阵（供 foundationpose_api.py --cam_K_file 使用）
    camera_intrinsics.json 相机内参详情
    rgb/    frame_000000.png ...
    depth/  frame_000000.png ...   （uint16，单位 mm）
    masks/  frame_000000.png ...   （uint8，255=螺母，0=背景）
    poses/  frame_000000.json ...  （关节角 + 末端位姿 + 螺母世界位姿，供核验用）
"""

import argparse
import json
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import mujoco.renderer as mj_renderer
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
XML_PATH    = os.path.join(os.path.dirname(__file__), "gen3_with_nut.xml")
CAMERA_NAME = "d435i_rgb_camera"
CAM_WIDTH   = 640
CAM_HEIGHT  = 480

JOINT_NAMES = ["joint_1", "joint_2", "joint_3",
               "joint_4", "joint_5", "joint_6", "joint_7"]

NUT_BODY_NAME = "square_nut"

HOME_QPOS = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]

# 深度有效范围（米）：超出该范围的像素（背景/远裁剪面）视为无效，写 0
DEPTH_MAX_M = 5.0

# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def get_camera_intrinsics(model, cam_name, width, height):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1:
        raise ValueError(f"找不到相机: {cam_name}")
    fovy_rad = model.cam_fovy[cam_id] * math.pi / 180.0
    fy = (height / 2.0) / math.tan(fovy_rad / 2.0)
    # MuJoCo 使用 square pixels：水平 FOV 由宽高比自动推导，fx == fy
    fx = fy
    return {
        "camera_matrix": [[fx, 0.0, width/2.0],
                          [0.0, fy, height/2.0],
                          [0.0, 0.0, 1.0]],
        "dist_coeffs": [[0.0, 0.0, 0.0, 0.0, 0.0]],
        "image_width": width,
        "image_height": height,
        "fovy_deg": float(model.cam_fovy[cam_id]),
        "note": "从 MuJoCo 仿真推算，无畸变",
    }


def get_gripper2base(model, data):
    base_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    bracelet_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bracelet_link")

    def make_T(pos, mat):
        T = np.eye(4)
        T[:3, :3] = mat.reshape(3, 3)
        T[:3, 3]  = pos
        return T

    T_base     = make_T(data.xpos[base_id],     data.xmat[base_id])
    T_bracelet = make_T(data.xpos[bracelet_id], data.xmat[bracelet_id])
    return np.linalg.inv(T_base) @ T_bracelet


def get_nut_pose_world(model, data):
    """返回螺母 body 在世界坐标系下的 4x4 位姿（仅供核验用的原始仿真真值）。"""
    nut_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, NUT_BODY_NAME)
    T = np.eye(4)
    T[:3, :3] = data.xmat[nut_id].reshape(3, 3)
    T[:3, 3]  = data.xpos[nut_id]
    return T


def get_nut_geom_ids(model):
    """获取螺母 body 下所有 geom 的 id 集合，用于从分割渲染结果中抠出掩码。"""
    nut_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, NUT_BODY_NAME)
    if nut_id == -1:
        raise ValueError(f"找不到 body: {NUT_BODY_NAME}")
    start = model.body_geomadr[nut_id]
    count = model.body_geomnum[nut_id]
    return set(range(start, start + count))


def apply_qpos(model, data, qpos_list):
    """直接设置关节角并前向运动学更新。"""
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                 for n in JOINT_NAMES]
    act_ids   = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                 for n in JOINT_NAMES]
    for jid, aid, q in zip(joint_ids, act_ids, qpos_list):
        addr = model.jnt_qposadr[jid]
        data.qpos[addr] = q
        data.qvel[addr] = 0.0
        data.ctrl[aid]  = q
    mujoco.mj_forward(model, data)


def settle_simulation(model, data, qpos_list, n_steps):
    """设置目标并仿真 n_steps 步稳定。"""
    apply_qpos(model, data, qpos_list)
    act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
               for n in JOINT_NAMES]
    for aid, q in zip(act_ids, qpos_list):
        data.ctrl[aid] = q
    for _ in range(n_steps):
        mujoco.mj_step(model, data)


def render_rgbd_mask(renderer, data, cam_opt, nut_geom_ids):
    """在同一个 update_scene() 之后依次渲染 RGB / 深度 / 分割掩码。

    返回:
      bgr_uint8   (H,W,3)  RGB→BGR，直接用于 cv2 保存/显示
      depth_m     (H,W)    float32，单位米，无效像素为 0
      mask_u8     (H,W)    uint8，255=螺母，0=背景
    """
    import cv2

    renderer.update_scene(data, camera=CAMERA_NAME, scene_option=cam_opt)

    rgb = renderer.render()
    bgr_uint8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    renderer.enable_depth_rendering()
    depth_m = renderer.render().copy()
    renderer.disable_depth_rendering()
    depth_m[(depth_m < 0.001) | (depth_m > DEPTH_MAX_M)] = 0.0

    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    obj_id   = seg[:, :, 0]
    obj_type = seg[:, :, 1]
    is_geom  = obj_type == mujoco.mjtObj.mjOBJ_GEOM
    nut_mask = is_geom & np.isin(obj_id, list(nut_geom_ids))
    mask_u8  = (nut_mask.astype(np.uint8)) * 255

    return bgr_uint8, depth_m, mask_u8


def save_frame(rgb_dir, depth_dir, masks_dir, poses_dir, idx,
               bgr_img, depth_m, mask_u8, qpos_arr, model, data):
    import cv2

    name = f"frame_{idx:06d}.png"
    cv2.imwrite(os.path.join(rgb_dir, name), bgr_img)

    depth_mm_u16 = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(depth_dir, name), depth_mm_u16)

    cv2.imwrite(os.path.join(masks_dir, name), mask_u8)

    T_g2b     = get_gripper2base(model, data)
    T_nut_wld = get_nut_pose_world(model, data)
    pose_name = f"frame_{idx:06d}.json"
    with open(os.path.join(poses_dir, pose_name), "w", encoding="utf-8") as f:
        json.dump({
            "index": idx,
            "joint_angles_rad": qpos_arr.tolist(),
            "gripper2base_4x4": T_g2b.tolist(),
            "nut_pose_world_4x4": T_nut_wld.tolist(),
        }, f, indent=4, ensure_ascii=False)
    print(f"  [保存 {idx:06d}] rgb/{name}  depth/{name}  masks/{name}  poses/{pose_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo 姿态估计（FoundationPose）数据采集：RGB + 深度 + 掩码",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xml",          default=XML_PATH)
    parser.add_argument("--data_dir",     default="pose_estimation_data")
    parser.add_argument("--settle_steps", type=int, default=300,
                        help="复现模式每个位姿的稳定步数")
    parser.add_argument("--preview",      action="store_true",
                        help="交互模式：键盘控制关节，SPACE 保存")
    parser.add_argument("--replay",       action="store_true",
                        help="复现模式：自动读取已保存位姿批量采集")
    args = parser.parse_args()

    print(f"[加载模型] {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)

    rgb_dir    = os.path.join(args.data_dir, "rgb")
    depth_dir  = os.path.join(args.data_dir, "depth")
    masks_dir  = os.path.join(args.data_dir, "masks")
    poses_dir  = os.path.join(args.data_dir, "poses")
    for d in (rgb_dir, depth_dir, masks_dir, poses_dir):
        os.makedirs(d, exist_ok=True)

    nut_geom_ids = get_nut_geom_ids(model)

    intrinsics = get_camera_intrinsics(model, CAMERA_NAME, CAM_WIDTH, CAM_HEIGHT)
    intr_path  = os.path.join(args.data_dir, "camera_intrinsics.json")
    with open(intr_path, "w", encoding="utf-8") as f:
        json.dump(intrinsics, f, indent=4, ensure_ascii=False)
    K = np.array(intrinsics["camera_matrix"])
    np.savetxt(os.path.join(args.data_dir, "cam_K.txt"), K, fmt="%.6f")
    print(f"[OK] 相机内参: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    renderer = mj_renderer.Renderer(model, height=CAM_HEIGHT, width=CAM_WIDTH)

    if args.replay:
        _run_replay(model, data, renderer, rgb_dir, depth_dir, masks_dir, poses_dir,
                    nut_geom_ids, args)
    else:
        _run_interactive(model, data, renderer, rgb_dir, depth_dir, masks_dir, poses_dir,
                         nut_geom_ids)

    renderer.close()


# ─────────────────────────────────────────────────────────────────────────────
# 交互模式
# ─────────────────────────────────────────────────────────────────────────────

def _run_interactive(model, data, renderer, rgb_dir, depth_dir, masks_dir, poses_dir,
                     nut_geom_ids):
    import cv2

    print("\n[交互模式]")
    print("  1~7    : 选择关节")
    print("  = / -  : 所选关节 +step / -step")
    print("  ] / [  : step×2 / step÷2")
    print("  r      : 重置 home 姿态")
    print("  SPACE  : 保存当前帧（RGB + 深度 + 掩码）")
    print("  q/ESC  : 退出\n")

    # 初始化 home 姿态
    qpos = np.array(HOME_QPOS, dtype=np.float64)
    apply_qpos(model, data, qpos)

    selected_joint = 0   # 0-based index
    step = 0.05          # 弧度
    saved_count = 0

    # 关节角限位（来自 XML actuator ctrlrange）
    joint_limits = [
        (-6.28, 6.28),   # joint_1
        (-2.25, 2.25),   # joint_2
        (-6.28, 6.28),   # joint_3
        (-2.58, 2.58),   # joint_4
        (-6.28, 6.28),   # joint_5
        (-2.10, 2.10),   # joint_6
        (-6.28, 6.28),   # joint_7
    ]

    # 渲染选项：隐藏 d435i 外壳（group5）
    cam_opt = mujoco.MjvOption()
    cam_opt.geomgroup[5] = 0

    # 全景相机
    renderer_pano = mj_renderer.Renderer(model, height=CAM_HEIGHT, width=CAM_WIDTH)
    pano_cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, pano_cam)
    pano_cam.lookat   = np.array([0.0, 0.3, 0.7])
    pano_cam.distance = 2.2
    pano_cam.azimuth  = -140.0
    pano_cam.elevation = -25.0

    # ── 鼠标拖拽状态（右侧全景面板） ────────────────────────────────────────
    WIN_NAME = "MuJoCo Pose-Est Collect  (left=camera  right=overview)"
    mouse = {
        "left_down":  False,   # 左键拖拽 → 旋转
        "right_down": False,   # 右键拖拽 → 平移 lookat
        "prev_x": 0,
        "prev_y": 0,
    }

    def _on_mouse(event, x, y, flags, param):
        # 只响应右半面板（x >= CAM_WIDTH）
        in_pano = x >= CAM_WIDTH

        if event == cv2.EVENT_LBUTTONDOWN and in_pano:
            mouse["left_down"] = True
            mouse["prev_x"], mouse["prev_y"] = x, y

        elif event == cv2.EVENT_RBUTTONDOWN and in_pano:
            mouse["right_down"] = True
            mouse["prev_x"], mouse["prev_y"] = x, y

        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            mouse["left_down"] = mouse["right_down"] = False

        elif event == cv2.EVENT_MOUSEMOVE:
            dx = x - mouse["prev_x"]
            dy = y - mouse["prev_y"]
            mouse["prev_x"], mouse["prev_y"] = x, y

            if mouse["left_down"] and in_pano:
                pano_cam.azimuth  += dx * 0.4
                pano_cam.elevation = float(
                    np.clip(pano_cam.elevation + dy * 0.3, -89.0, 89.0))

            elif mouse["right_down"] and in_pano:
                yaw_rad = math.radians(pano_cam.azimuth)
                scale   = pano_cam.distance * 0.001
                pano_cam.lookat[0] += (-math.sin(yaw_rad) * dx
                                       + math.cos(yaw_rad) * 0) * scale
                pano_cam.lookat[1] += ( math.cos(yaw_rad) * dx
                                       + math.sin(yaw_rad) * 0) * scale
                pano_cam.lookat[2] -= dy * scale

        elif event == cv2.EVENT_MOUSEWHEEL and in_pano:
            if flags > 0:
                pano_cam.distance = max(pano_cam.distance * 0.9, 0.2)
            else:
                pano_cam.distance = min(pano_cam.distance * 1.1, 10.0)

    def render_frame():
        # 相机视角（纯净帧，用于保存）
        renderer.update_scene(data, camera=CAMERA_NAME, scene_option=cam_opt)
        bgr_cam_clean = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

        # 全景视角
        renderer_pano.update_scene(data, camera=pano_cam)
        bgr_pano = cv2.cvtColor(renderer_pano.render(), cv2.COLOR_RGB2BGR)

        # HUD 叠加在显示副本上（不污染保存的原图）
        bgr_cam_display = bgr_cam_clean.copy()
        lines_cam = [
            f"[Camera]  Saved: {saved_count}",
            f"Joint {selected_joint+1} selected  step={step:.3f}rad",
            "SPACE:save  R:reset  Q:quit",
        ]
        for i, txt in enumerate(lines_cam):
            cv2.putText(bgr_cam_display, txt, (8, 24 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 220, 0) if i < 2 else (0, 200, 200), 1)

        # HUD：全景视角，显示各关节角
        cv2.putText(bgr_pano, "[Overview] LDrag:rotate  RDrag:pan  Wheel:zoom",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 255), 1)
        for ji, q in enumerate(qpos):
            mark = ">>>" if ji == selected_joint else "   "
            txt  = f"{mark} J{ji+1}: {math.degrees(q):+7.2f} deg"
            color = (0, 255, 100) if ji == selected_joint else (200, 200, 200)
            cv2.putText(bgr_pano, txt, (8, 45 + ji * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)

        return np.hstack([bgr_cam_display, bgr_pano])

    cv2.namedWindow(WIN_NAME)
    cv2.setMouseCallback(WIN_NAME, _on_mouse)

    while True:
        combined = render_frame()
        cv2.imshow(WIN_NAME, combined)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            break

        elif ord("1") <= key <= ord("7"):
            selected_joint = key - ord("1")

        elif key in (ord("="), ord("+")):
            lo, hi = joint_limits[selected_joint]
            qpos[selected_joint] = np.clip(qpos[selected_joint] + step, lo, hi)
            apply_qpos(model, data, qpos)

        elif key == ord("-"):
            lo, hi = joint_limits[selected_joint]
            qpos[selected_joint] = np.clip(qpos[selected_joint] - step, lo, hi)
            apply_qpos(model, data, qpos)

        elif key == ord("]"):
            step = min(step * 2, 0.5)

        elif key == ord("["):
            step = max(step / 2, 0.005)

        elif key == ord("r"):
            qpos = np.array(HOME_QPOS, dtype=np.float64)
            apply_qpos(model, data, qpos)
            print("  [重置] home 姿态")

        elif key == ord(" "):
            bgr, depth_m, mask_u8 = render_rgbd_mask(renderer, data, cam_opt, nut_geom_ids)
            save_frame(rgb_dir, depth_dir, masks_dir, poses_dir, saved_count,
                       bgr, depth_m, mask_u8, qpos, model, data)
            saved_count += 1

    cv2.destroyAllWindows()
    renderer_pano.close()
    print(f"\n[退出] 共保存 {saved_count} 组数据到 '{os.path.dirname(rgb_dir)}/'")
    if saved_count > 0:
        data_dir = os.path.dirname(rgb_dir)
        print(f"  复现命令: python {os.path.basename(__file__)} --replay --data_dir {data_dir}")
        print(f"  姿态估计: python foundationpose_api.py --mesh <nut_mesh> "
              f"--input {data_dir} --cam_K_file {data_dir}/cam_K.txt")


# ─────────────────────────────────────────────────────────────────────────────
# 复现模式
# ─────────────────────────────────────────────────────────────────────────────

def _run_replay(model, data, renderer, rgb_dir, depth_dir, masks_dir, poses_dir,
                nut_geom_ids, args):
    import glob

    pose_files = sorted(glob.glob(os.path.join(poses_dir, "frame_*.json")))
    if not pose_files:
        print(f"[错误] 在 {poses_dir} 中未找到位姿文件，请先运行交互模式保存数据。")
        return

    print(f"\n[复现模式] 找到 {len(pose_files)} 个位姿，逐一复现采集...")

    cam_opt = mujoco.MjvOption()
    cam_opt.geomgroup[5] = 0

    saved = 0
    for pose_path in pose_files:
        with open(pose_path, encoding="utf-8") as f:
            pose_data = json.load(f)

        idx      = pose_data["index"]
        qpos_arr = np.array(pose_data["joint_angles_rad"], dtype=np.float64)

        settle_simulation(model, data, qpos_arr, args.settle_steps)

        bgr, depth_m, mask_u8 = render_rgbd_mask(renderer, data, cam_opt, nut_geom_ids)
        save_frame(rgb_dir, depth_dir, masks_dir, poses_dir, idx,
                   bgr, depth_m, mask_u8, qpos_arr, model, data)
        saved += 1

    print(f"\n[完成] 共复现采集 {saved} 组数据")
    data_dir = os.path.dirname(rgb_dir)
    print(f"  姿态估计: python foundationpose_api.py --mesh <nut_mesh> "
          f"--input {data_dir} --cam_K_file {data_dir}/cam_K.txt")


if __name__ == "__main__":
    main()
