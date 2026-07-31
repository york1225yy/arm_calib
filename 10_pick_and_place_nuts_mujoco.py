#!/usr/bin/env python3
"""
10_pick_and_place_nuts_mujoco.py
---------------------------------
Kinova Gen3 + 桌面固定俯视 D435i 相机 demo：感知（FoundationPose 位姿估计）
+ 抓取规划控制（IK + 运动队列）+ 键盘手动控制机械臂 —— 仅包含 UI/编排层代码。

代码拆分说明：本文件只保留"非算法"的编排/UI 代码（argparse、MuJoCo 模型
加载、OpenCV 显示、键盘状态机、录像、viewer）。两块核心算法被拆分到独立
文件中，统一放在 pick_place_modules/ 目录下便于查找，也便于两名工程师
分别独立开发：
  - pick_place_modules/sim_common.py      ：共享的非算法基础设施（MuJoCo
                         读写原语、坐标变换、标定文件加载、3D 可视化绘制
                         工具）。
  - pick_place_modules/nut_perception.py  ：感知算法（NutPerceptionModule
                         封装 foundationpose_api.MultiObjectPoseEstimator，
                         并把螺母位姿换算成机械臂基坐标系下的抓取/法兰
                         目标位姿）。foundationpose_api.py 本身已经是感知
                         底层 API，这里不重复实现，只做业务封装。
  - pick_place_modules/grasp_planner.py   ：抓取规划控制算法
                         （solve_ik_position() + MotionQueuePlanner：构建
                         "去程/回程"运动队列，并逐帧插值输出关节角）。
本文件中的 Demo 类只是把三者"粘合"在一起，不包含任何感知或
IK/运动规划算法的具体实现。

按键说明：
  'e' 开始位姿估计+跟踪   'r' 重新初始化   'p' 打印6D目标位姿
  'g' 依次运动到每个螺母的法兰目标位置再返回   'q' 退出
  '1'-'7' 选中关节   '['/']' 关节角减/增   'c'/'o' 夹爪闭合/张开

用法：
  python 10_pick_and_place_nuts_mujoco.py --viewer
  python 10_pick_and_place_nuts_mujoco.py --no_gui --auto --save_video output_3nuts/pose_estimation_vision.mp4
"""

import argparse
import math
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pick_place_modules"))
from sim_common import (
    ARM_JOINT_NAMES, BASE_BODY_NAME, FLANGE_TARGET_BBOX,
    NUT_BODY_NAMES, TOP_CAMERA_NAME, draw_posed_3d_box_simple, draw_xyz_axis_simple,
    get_all_nut_geom_ids, get_camera_intrinsics, load_start_qpos, load_tcp_flange,
    project_3d_to_2d, render_rgb_depth_masks, set_arm_ctrl, set_arm_qpos,
    set_gripper_ctrl_ratio,
)
from nut_perception import NutPerceptionModule
from grasp_planner import MotionQueuePlanner

MANUAL_JOINT_STEP_DEG = 2.0    # 每次按 '['/']' 调整关节角度的步长（度）
MANUAL_GRIPPER_STEP = 0.05     # 每次按 'c'/'o' 调整夹爪开合比例的步长（0~1）


class Demo:
    def __init__(self, args):
        self.args = args
        here = os.path.dirname(os.path.abspath(args.xml)) or "."
        self.here = here

        print(f"[load model] {args.xml}")
        self.model = mujoco.MjModel.from_xml_path(args.xml)
        # 关闭离屏渲染的 MSAA（多重采样抗锯齿）：开启 MSAA 时，分割渲染出的
        # 整数 ID 会在物体边缘被插值混合成"垃圾值"，从而破坏用于
        # FoundationPose 平移初值估计的掩码外接矩形。offsamples=0 可以
        # 保证分割 ID 精确不失真。
        self.model.vis.quality.offsamples = 0

        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.q_start = load_start_qpos(args.start_pose_file)
        set_arm_ctrl(self.model, self.data, self.q_start)
        set_arm_qpos(self.model, self.data, self.q_start)
        set_gripper_ctrl_ratio(self.model, self.data, 0.0)
        mujoco.mj_forward(self.model, self.data)

        T_tcp_flange = load_tcp_flange(args.tcp_flange_file)
        grasp_pose_files = {
            "square_nut": args.grasp_pose_file,
            "square_nut_2": args.grasp_pose_file,
            "round_nut": args.round_grasp_pose_file,
        }
        mesh_files = {
            "square_nut": args.mesh,
            "square_nut_2": args.mesh,
            "round_nut": args.round_mesh,
        }
        self.nut_order = [n.strip() for n in args.nut_order.split(",") if n.strip()]

        self.nut_geom_ids = get_all_nut_geom_ids(self.model, NUT_BODY_NAMES)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.K = get_camera_intrinsics(self.model, TOP_CAMERA_NAME, 640, 480)
        print(f"[top camera intrinsics]\n{self.K}")

        # 感知算法模块——具体实现见 nut_perception.py
        self.perception = NutPerceptionModule(
            self.model, self.data,
            top_camera_name=TOP_CAMERA_NAME, base_body_name=BASE_BODY_NAME,
            mesh_files=mesh_files, weights_dir=args.weights_dir,
            grasp_pose_files=grasp_pose_files, tcp_flange=T_tcp_flange, K=self.K,
            here=here, est_refine_iter=args.est_refine_iter,
            track_refine_iter=args.track_refine_iter,
        )

        self.state = "idle"  # idle -> tracking（手动控制随时可用，不受此状态影响）
        self._just_entered_tracking = False

        self.video_writer = None
        if args.save_video:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_video)) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(args.save_video, fourcc, args.fps, (640, 480))
            print(f"[video] saving to {args.save_video} (640x480 @ {args.fps}fps)")

        self.viewer = None
        if args.viewer:
            import mujoco.viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.model, self.data)

        self.window_name = "Top D435i Camera - FoundationPose 6D Pose"
        self.frame_idx = 0
        self._quit = False

        # 手动控制相关状态
        self.active_joint_idx = 0
        self.q_manual = self.q_start.copy()
        self.gripper_ratio = 0.0

        # 抓取规划控制算法模块——具体实现见 grasp_planner.py。
        # ik_data 是与真实仿真 self.data 相互独立的 scratch MjData，
        # 只用于反复调用 mj_forward 做 IK 试算，不会影响真实物理状态。
        self.ik_data = mujoco.MjData(self.model)
        self.planner = MotionQueuePlanner(
            self.model, self.ik_data, self.q_start,
            move_seconds=args.move_seconds, hold_seconds=args.hold_seconds, fps=args.fps,
        )

    def render_and_display(self, status_lines=None):
        rgb, depth_m, masks = render_rgb_depth_masks(
            self.renderer, self.data, TOP_CAMERA_NAME, self.nut_geom_ids)

        waiting_text = None
        if self.state == "tracking":
            if self.perception.is_busy:
                waiting_text = "Estimating pose... please wait (first time may take up to ~1 min to load weights)"
            elif self._just_entered_tracking:
                if not self.perception.has_pending_async:
                    self.perception.start_estimate_async(rgb, masks, depth_m)
                    waiting_text = "Starting pose estimation..."
                else:
                    self.perception.poll_async_result()
                    self._just_entered_tracking = False
            else:
                self.perception.track(rgb, depth_m)

            vis_bgr = self.perception.visualize_all(rgb)
            vis_bgr = self._draw_flange_targets(vis_bgr)
            if waiting_text:
                cv2.putText(vis_bgr, waiting_text, (8, 460), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis_bgr, waiting_text, (8, 460), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            vis_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        lines = list(status_lines or [])
        active_joint_name = ARM_JOINT_NAMES[self.active_joint_idx]
        lines.append(f"state={self.state}  tracked={list(self.perception.latest_poses.keys())}")
        lines.append(f"[manual] active_joint={active_joint_name} ({self.active_joint_idx + 1}/7) "
                     f"angle={math.degrees(self.q_manual[self.active_joint_idx]):.1f}deg  "
                     f"gripper={self.gripper_ratio:.2f}")
        lines.append("[1-7] select joint  [ [/] ] +-joint  [c] close gripper  [o] open gripper")
        if self.state == "idle":
            lines.append("[e] start pose estimation   [q] quit")
        elif self.state == "tracking":
            ready = all(n in self.perception.latest_poses for n in self.nut_order)
            lines.append(f"[p] {'print 6D grasp pose (ready)' if ready else 'print 6D grasp pose (waiting for all nuts)'}   [r] reinit   [q] quit")
            if not self.planner.is_idle:
                lines.append(f"[G] running (queue remaining={self.planner.queue_remaining})")
            else:
                lines.append(f"[g] {'go to each target & back (ready)' if ready else 'go to each target & back (waiting for all nuts)'}")
        for i, line in enumerate(lines):
            cv2.putText(vis_bgr, line, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis_bgr, line, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if not self.args.no_gui:
            cv2.imshow(self.window_name, vis_bgr)
        if self.video_writer is not None:
            self.video_writer.write(vis_bgr)

        key = self._poll_key()
        self.frame_idx += 1
        return key

    def _poll_key(self):
        if self.args.no_gui:
            return -1
        k = cv2.waitKey(1) & 0xFF
        return k if k != 255 else -1

    def _draw_flange_targets(self, vis_bgr):
        # 把每个已估计出位姿的螺母对应的法兰目标位置画成绿色 3D 框+坐标轴，
        # 便于直观对照"螺母在哪 / 法兰要去哪"。坐标变换交给 self.perception 完成。
        if not self.perception.latest_poses:
            return vis_bgr
        T_base_cam = np.linalg.inv(self.perception.T_cam_base)  # base坐标系 -> 相机坐标系
        for name in self.perception.latest_poses:
            T_flange_base_est = self.perception.compute_T_flange_base(name)
            T_flange_cam = T_base_cam @ T_flange_base_est
            if T_flange_cam[2, 3] <= 1e-6:
                continue  # 目标在相机背后/重合，跳过（避免除零/投影到画面外）
            vis_bgr = draw_posed_3d_box_simple(self.K, vis_bgr, T_flange_cam,
                                                FLANGE_TARGET_BBOX, line_color=(0, 255, 0))
            vis_bgr = draw_xyz_axis_simple(vis_bgr, T_flange_cam, self.K, scale=0.06)
            u, v = project_3d_to_2d(np.array([0., 0., 0., 1.]), self.K, T_flange_cam)
            cv2.putText(vis_bgr, f"flange<-{name}", (int(u) + 6, int(v) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2, cv2.LINE_AA)
        return vis_bgr

    # 手动关节/夹爪控制（idle/tracking 任意状态下都可用）
    def _apply_manual_joint_delta(self, delta_rad):
        idx = self.active_joint_idx
        self.q_manual[idx] += delta_rad
        set_arm_ctrl(self.model, self.data, self.q_manual)
        print(f"    [manual] {ARM_JOINT_NAMES[idx]} -> {math.degrees(self.q_manual[idx]):.1f}deg")

    def _apply_manual_gripper_delta(self, delta_ratio):
        self.gripper_ratio = float(np.clip(self.gripper_ratio + delta_ratio, 0.0, 1.0))
        set_gripper_ctrl_ratio(self.model, self.data, self.gripper_ratio)
        print(f"    [manual] gripper ratio -> {self.gripper_ratio:.2f} (0=open, 1=closed)")

    def handle_key(self, key):
        if key in (ord('q'), ord('Q')):
            self._quit = True
        elif key in (ord('e'), ord('E')) and self.state == "idle":
            print("[state] idle -> tracking (start pose estimation + tracking)")
            self.state = "tracking"
            self._just_entered_tracking = True
        elif key in (ord('r'), ord('R')) and self.state == "tracking":
            if self.perception.is_busy:
                print("[info] pose estimation still running in background, wait before reinit.")
            else:
                print("[state] reinit pose estimation")
                self.perception.clear()
                self._just_entered_tracking = True
        elif key in (ord('p'), ord('P')) and self.state == "tracking":
            if self.perception.is_busy:
                print("[info] pose estimation still running, wait before printing 6D pose.")
            elif all(n in self.perception.latest_poses for n in self.nut_order):
                self.compute_and_print_final_poses()
            else:
                print("[info] not all nuts have an estimated pose yet.")
        elif key in (ord('g'), ord('G')) and self.state == "tracking":
            if self.perception.is_busy:
                print("[info] pose estimation still running, wait before pressing 'G'.")
            else:
                self.build_motion_queue()
        elif key in tuple(ord(str(d)) for d in range(1, 8)):
            self.active_joint_idx = int(chr(key)) - 1
            print(f"    [manual] selected joint {ARM_JOINT_NAMES[self.active_joint_idx]}")
        elif key == ord('['):
            self._apply_manual_joint_delta(-math.radians(MANUAL_JOINT_STEP_DEG))
        elif key == ord(']'):
            self._apply_manual_joint_delta(math.radians(MANUAL_JOINT_STEP_DEG))
        elif key in (ord('c'), ord('C')):
            self._apply_manual_gripper_delta(MANUAL_GRIPPER_STEP)
        elif key in (ord('o'), ord('O')):
            self._apply_manual_gripper_delta(-MANUAL_GRIPPER_STEP)

    def settle(self):
        # 让物理仿真先运行一段时间，使螺母在重力作用下自然落稳到桌面上
        print(f"[settle] running {self.args.settle_steps} physics steps so nuts settle under gravity ...")
        display_every = max(1, round((1.0 / self.args.fps) / self.model.opt.timestep))
        for step in range(self.args.settle_steps):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()
            if step % display_every == 0:
                key = self.render_and_display(status_lines=["settling physics ..."])
                self.handle_key(key)
                if self._quit:
                    return

    def compute_and_print_final_poses(self):
        # 'p' 键：打印每个螺母对应的最终机械臂6D抓取/法兰目标位姿（不驱动机械臂）。
        # 坐标变换全部委托给 self.perception（感知算法模块）完成。
        print(f"\n{'=' * 60}\n[6D pose] final grasp/flange target poses: {self.nut_order}\n{'=' * 60}")
        for name in self.nut_order:
            T_grasp_base_est, T_nut_base_est = self.perception.compute_T_grasp_base(name)
            T_flange_base_est = self.perception.compute_T_flange_base(name)
            _, pos_err, rot_err_deg = self.perception.compute_pose_error_vs_ground_truth(
                self.model, self.data, name)

            grasp_xyz = T_grasp_base_est[:3, 3]
            grasp_rpy_deg = Rotation.from_matrix(T_grasp_base_est[:3, :3]).as_euler("xyz", degrees=True)
            grasp_quat_xyzw = Rotation.from_matrix(T_grasp_base_est[:3, :3]).as_quat()
            flange_xyz = T_flange_base_est[:3, 3]
            flange_rpy_deg = Rotation.from_matrix(T_flange_base_est[:3, :3]).as_euler("xyz", degrees=True)

            print(f"\n[{name}]")
            print(f"  [check] pos_err={pos_err * 1000:.1f}mm  rot_err={rot_err_deg:.1f}deg (vs sim ground truth)")
            print(f"  grasp TCP 6D pose (base frame): xyz(m)={grasp_xyz}  rpy(deg)={grasp_rpy_deg}  "
                  f"quat(xyzw)={grasp_quat_xyzw}")
            print(f"  flange 6D pose (base frame): xyz(m)={flange_xyz}  rpy(deg)={flange_rpy_deg}")
            print(f"  T_grasp_base=\n{T_grasp_base_est}")
            print(f"  T_flange_base=\n{T_flange_base_est}")

    def build_motion_queue(self):
        # 'g' 键：为每个已估计出位姿的螺母求解位置IK并规划"去程/回程"运动队列。
        # IK求解+队列管理全部委托给 self.planner（抓取规划控制算法模块），
        # 本方法只负责从 self.perception 取出每个螺母的目标位置。
        if not self.planner.is_idle:
            print("[info] previous 'G' motion still running, wait until it finishes.")
            return
        if not all(n in self.perception.latest_poses for n in self.nut_order):
            print("[info] not all nuts have an estimated pose yet.")
            return

        print(f"\n{'=' * 60}\n[G] solving position IK and building motion queue: {self.nut_order}\n{'=' * 60}")
        targets = {name: self.perception.compute_T_flange_base(name)[:3, 3] for name in self.nut_order}
        ik_solutions = self.planner.build_queue(targets, q_seed=self.q_manual)
        for name, q_ik in ik_solutions.items():
            print(f"  [{name}] flange target xyz(m)={targets[name]}  IK solution: {np.degrees(q_ik).round(1)} deg")

    def update_motion(self):
        # 每个显示帧调用一次：若运动队列非空，取出本帧插值后的新关节角，
        # 更新 q_manual 并下发 ctrl（真实物理持续 mj_step，手臂会平滑运动过去）。
        q_new = self.planner.update(self.q_manual)
        if q_new is not None:
            self.q_manual = q_new
            set_arm_ctrl(self.model, self.data, self.q_manual)

    def run(self):
        # 主循环：idle/tracking 状态下等待按键；手动控制/运动队列随时生效
        self.settle()
        display_every = max(1, round((1.0 / self.args.fps) / self.model.opt.timestep))
        auto_frame_counter = 0
        while not self._quit:
            self.update_motion()
            for _ in range(display_every):
                mujoco.mj_step(self.model, self.data)
                if self.viewer is not None:
                    self.viewer.sync()

            key = self.render_and_display()
            self.handle_key(key)

            if self.args.auto:
                # 自动化/无人值守模式：模拟按键序列，用于无显示环境下的测试
                auto_frame_counter += 1
                if self.state == "idle" and auto_frame_counter >= self.args.auto_estimate_after:
                    self.handle_key(ord('e'))
                elif self.state == "tracking" and auto_frame_counter >= (
                        self.args.auto_estimate_after + self.args.auto_print_after):
                    if all(n in self.perception.latest_poses for n in self.nut_order):
                        self.compute_and_print_final_poses()
                        self._quit = True

            if not self.args.no_gui:
                time.sleep(0.0)  # 让出时间片，cv2.waitKey 已在 render_and_display 中处理节流

        self.cleanup()

    def cleanup(self):
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"\n[video] saved to {self.args.save_video}")
        if self.viewer is not None:
            self.viewer.close()
        if not self.args.no_gui:
            cv2.destroyAllWindows()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Gen3 + top-down D435i camera: FoundationPose estimation + IK planning + manual keyboard control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xml", default=os.path.join(here, "gen3_with_gripper_and_nuts.xml"))
    parser.add_argument("--start_pose_file",
                        default=os.path.join(here, "pose_estimation_data_3nuts", "poses", "frame_000000.json"))
    parser.add_argument("--mesh", default=os.path.join(here, "nut_mesh", "textured_simple.obj"),
                        help="shared mesh for square_nut/square_nut_2")
    parser.add_argument("--round_mesh", default=os.path.join(here, "nut_mesh", "round_nut_textured_simple.obj"),
                        help="mesh for round_nut")
    parser.add_argument("--grasp_pose_file", default=os.path.join(here, "nut_grasp_pose.json"))
    parser.add_argument("--round_grasp_pose_file", default=os.path.join(here, "nut_grasp_pose_round.json"))
    parser.add_argument("--tcp_flange_file", default=os.path.join(here, "tcp_flange.json"))
    parser.add_argument("--weights_dir", default=os.path.join(here, "weights"))
    parser.add_argument("--nut_order", default="square_nut,square_nut_2,round_nut")

    parser.add_argument("--settle_steps", type=int, default=800,
                        help="physics steps to run at startup so nuts settle onto the table")
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--move_seconds", type=float, default=2.5,
                        help="duration of each go/return interpolated motion segment triggered by 'G'")
    parser.add_argument("--hold_seconds", type=float, default=1.0,
                        help="seconds to hold at each target/start pose before continuing")

    parser.add_argument("--fps", type=int, default=30, help="display/recording frame rate")
    parser.add_argument("--save_video", default=None, help="path to save top-camera mp4 (with pose overlay)")
    parser.add_argument("--viewer", action="store_true", help="also open a third-person MuJoCo viewer window")
    parser.add_argument("--no_gui", action="store_true",
                        help="do not open the OpenCV camera window (headless, use with --auto)")
    parser.add_argument("--auto", action="store_true",
                        help="auto-simulate keypresses ('e' then print 6D pose) for headless testing")
    parser.add_argument("--auto_estimate_after", type=int, default=30,
                        help="in --auto mode, frames to wait before auto-pressing 'e'")
    parser.add_argument("--auto_print_after", type=int, default=60,
                        help="in --auto mode, frames after entering tracking before auto-printing 6D pose")
    args = parser.parse_args()

    if not args.no_gui:
        os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "glfw")

    demo = Demo(args)
    demo.run()


if __name__ == "__main__":
    main()
