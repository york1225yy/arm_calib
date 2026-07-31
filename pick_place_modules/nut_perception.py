#!/usr/bin/env python3
"""
nut_perception.py
------------------
【感知算法模块】——从 10_pick_and_place_nuts_mujoco.py 中拆分出来的"感知"
部分：调用 foundationpose_api.MultiObjectPoseEstimator 得到每个螺母在
相机坐标系下的 6D 位姿，并把该位姿换算到机械臂基坐标系、进一步结合抓取
标定偏移得到"抓取目标位姿"和"末端法兰目标位姿"。

设计目的
--------
真正的 FoundationPose 底层推理已经封装成独立的 API 文件
foundationpose_api.py（题目要求"感知部分已独立成 API，不需要额外的 API
代码"），因此本文件不重新实现任何 FoundationPose 相关的底层推理逻辑，
只做"感知算法"这一层的业务封装：
  - 管理 MultiObjectPoseEstimator 的生命周期（懒加载、异步初始化避免
    阻塞 UI 线程、连续帧 track_all 增量跟踪）；
  - 对估计结果做鲁棒性过滤（掩码像素数校验、深度合理性校验、薄片对称
    物体上下翻转歧义修正）；
  - 把相机坐标系下的螺母位姿，通过坐标变换链路换算成机械臂基坐标系下的
    "抓取目标位姿"(T_grasp_base) 与"法兰目标位姿"(T_flange_base)。

本文件只依赖 sim_common.py 里的纯几何/加载工具函数与
foundationpose_api.MultiObjectPoseEstimator，不包含任何 IK 求解、运动
队列等"抓取规划控制"逻辑（那些在 grasp_planner.py 中）。这样只做感知
算法开发的同事，只需要修改本文件（以及必要时 foundationpose_api.py），
完全不需要接触 UI/渲染代码（10_pick_and_place_nuts_mujoco.py）或运动
规划代码（grasp_planner.py）。
"""

import os
import threading

import numpy as np
from scipy.spatial.transform import Rotation

from foundationpose_api import MultiObjectPoseEstimator
from sim_common import get_body_T_world, get_cam_T_world_cv, load_grasp_pose, rotation_log

# ── 位姿估计鲁棒性过滤（解决"没有目标的位置也出现目标"的异常检测问题）───
# 螺母分割掩码有效像素数低于此阈值时，视为"当前帧看不清/被遮挡"，直接跳过
# 本帧对该目标的 estimate_all() 调用（避免用一小撮噪声像素去初始化出一个
# 完全错误的位姿——mask 太小/太碎时 FoundationPose 的 PnP/优化很容易收敛到
# 错误极值，导致包围框出现在画面中根本没有物体的位置）。
MIN_MASK_PIXELS = 300
# 俯视相机固定安装高度（xml 中 top_d435i_mount pos z=1.6）减去桌面高度
# （table 上表面 z=0.78）得到的名义工作距离，用于合理性校验：一个真正贴在
# 桌面上的螺母，其在相机坐标系下的 Z（深度，米）应该落在下方区间内；明显
# 超出此区间的估计结果视为跟踪/估计出现异常（例如收敛到了错误的局部最优、
# 或目标被遮挡后 track() 仍在对着错误的背景像素做增量跟踪），应被丢弃而不是
# 继续显示/参与后续的坐标变换计算。
CAM_HEIGHT_ABOVE_TABLE_M = 1.6 - 0.78
DEPTH_SANITY_RANGE_M = (CAM_HEIGHT_ABOVE_TABLE_M - 0.25, CAM_HEIGHT_ABOVE_TABLE_M + 0.25)


def filter_valid_masks(masks, min_pixels=MIN_MASK_PIXELS):
    """丢弃像素数过少（噪声/严重遮挡）的掩码，避免用碎片掩码去初始化位姿
    估计而产生完全错误的结果。返回值可直接传给 estimate_all()。"""
    valid = {}
    for name, mask in masks.items():
        if mask is not None and int(np.count_nonzero(mask)) >= min_pixels:
            valid[name] = mask
    return valid


def pose_is_plausible(T_nut_cam, depth_range_m=DEPTH_SANITY_RANGE_M):
    """粗略合理性校验：一个真正贴在桌面上的螺母，其在相机坐标系下的 Z（深度）
    应落在"俯视相机安装高度 ± 容差"范围内。用于过滤 FoundationPose 估计/
    跟踪偶尔收敛到错误位置（例如跟丢后对着背景像素继续跟踪）产生的"幽灵"
    位姿——这类异常位姿的深度往往会明显偏离桌面所在的深度范围。"""
    if T_nut_cam is None:
        return False
    z_cam = float(T_nut_cam[2, 3])
    return depth_range_m[0] <= z_cam <= depth_range_m[1]


def fix_flat_object_updown_ambiguity(T_nut_base):
    """修正"薄片状对称螺母"常见的 180° 上下翻转歧义。

    背景（"机械臂抓取位置根本不是 nut 真实位置"问题的根因）：
    通过对比 FoundationPose 视觉估计位姿与仿真真值位姿（同一 base_link
    坐标系下）发现：位置（平移）估计误差始终 <1.5mm，完全正确；但
    square_nut / square_nut_2 的旋转误差高达 ~180°（round_nut 因绕通孔轴
    旋转对称而没有这个问题，误差仅 0.8°）。

    原因：square_nut/square_nut_2 是薄片状物体（沿局部 Z 轴很薄），从俯视
    相机（几乎垂直向下）看过去，"正面朝上"和"整体绕水平轴翻转 180°（即
    正面朝下、背面朝上）"两种姿态在轮廓/深度图上几乎完全一样 —— 这是薄片
    对称物体本身固有的姿态歧义，不是坐标变换或坐标系定义的 bug（平移误差
    <1.5mm 已经证明整条坐标变换链路 T_cam_base/T_nut_cam/矩阵乘法 都是对的）。

    该歧义体现为：估计出的螺母局部 Z 轴（螺母厚度方向法线，nut_grasp_pose.json
    中 T_grasp_nut 的抓取偏移正是按"局部 +Z 朝上"这个约定标定的）在世界坐标系
    下指向下方（穿入桌面）而不是上方——这在物理上不可能（螺母平放在桌面上，
    法线必然朝上），是一个可以事后按几何规则直接修正的、明确的错误分支，
    而不需要修改感知模型本身。

    修正方法：若估计位姿的局部 Z 轴（旋转矩阵第 3 列）在世界系下的 Z 分量
    为负（朝下），说明命中了"翻转过来"的这一支解，绕该位姿的局部 X 轴
    旋转 180°，把 Z 轴重新翻回朝上，即可恢复与标定文件一致的"正面朝上"
    约定，抓取偏移 T_grasp_nut 就能正确复合到把手上真正所在的位置。
    """
    R = T_nut_base[:3, :3]
    local_z_in_world = R[:, 2]
    if local_z_in_world[2] < 0.0:
        R_flip_x = Rotation.from_rotvec(np.array([np.pi, 0.0, 0.0])).as_matrix()
        T_fixed = T_nut_base.copy()
        T_fixed[:3, :3] = R @ R_flip_x
        return T_fixed
    return T_nut_base


class NutPerceptionModule:
    """封装"多螺母 6D 位姿估计 + 坐标变换到机械臂基坐标系 + 抓取/法兰目标
    位姿计算"的感知算法模块。

    典型用法（详见 10_pick_and_place_nuts_mujoco.py 中 Demo 类的调用方式）：
        perception = NutPerceptionModule(
            model, data, top_camera_name=TOP_CAMERA_NAME,
            mesh_files={...}, weights_dir=..., grasp_pose_files={...},
            tcp_flange_file=...)
        ...
        perception.start_estimate_async(rgb, masks, depth_m)   # 首帧异步初始化
        ...
        poses = perception.track(rgb, depth_m)                 # 后续帧增量跟踪
        ...
        T_grasp_base, T_nut_base = perception.compute_T_grasp_base(name)
        T_flange_base = perception.compute_T_flange_base(name)
    """

    def __init__(self, model, data, *, top_camera_name, base_body_name,
                 mesh_files, weights_dir, grasp_pose_files, tcp_flange, K,
                 here=".", est_refine_iter=5, track_refine_iter=2,
                 debug_dir="output_vision"):
        self.model = model
        self.mesh_files = mesh_files
        self.weights_dir = weights_dir
        self.grasp_pose_files = grasp_pose_files
        self.T_tcp_flange = tcp_flange
        self.K = K
        self.here = here
        self.est_refine_iter = est_refine_iter
        self.track_refine_iter = track_refine_iter
        self.debug_dir = debug_dir

        # T_cam_base：俯视相机相对机械臂基座的固定位姿（与关节角无关，因为
        # 相机和 base_link 都是固定在世界坐标系中的静态 body），一次性算出。
        T_base_world = get_body_T_world(model, data, base_body_name)
        T_cam_world_cv = get_cam_T_world_cv(model, data, top_camera_name)
        self.T_cam_base = np.linalg.inv(T_base_world) @ T_cam_world_cv
        self.T_base_world = T_base_world

        self.estimator = None  # 首次异步估计时才加载权重（懒加载）
        self.latest_poses = {}  # name -> T_nut_cam (4x4)，最近一次估计/跟踪结果

        # ── 异步推理相关状态（解决"按 E 之后窗口卡死/系统提示无响应"问题）──
        # 首次加载 FoundationPose 权重 + 首帧 estimate_all() register 通常需要
        # 数秒到一分钟不等，如果在主线程里同步阻塞执行，期间既不刷新画面也
        # 不处理 UI 消息循环，操作系统会认为窗口"未响应"。这里把"权重加载 +
        # estimate_all()"这一耗时操作放到后台线程执行，调用方（UI 主循环）
        # 只需轮询 is_busy / poll_async_result() 即可。逐帧 track() 本身已经
        # 是为实时调用设计的（较快），因此不需要异步化，仍同步调用。
        self._infer_thread = None
        self._infer_busy = False
        self._infer_result = None
        self._infer_error = None

    # ── 生命周期管理 ────────────────────────────────────────────────────
    @property
    def is_busy(self):
        return self._infer_busy

    def reset(self, name):
        self.latest_poses.pop(name, None)
        if self.estimator is not None:
            self.estimator.reset(name)

    def clear(self):
        self.latest_poses.clear()
        self._infer_thread = None

    # ── 首帧异步初始化（estimate_all，需要掩码）───────────────────────────
    def start_estimate_async(self, rgb, masks, depth_m):
        self._infer_busy = True
        self._infer_result = None
        self._infer_error = None

        def worker(rgb=rgb, masks=masks, depth_m=depth_m):
            try:
                if self.estimator is None:
                    print("[FoundationPose] 首次使用，正在加载权重（较慢，请稍候）…")
                    self.estimator = MultiObjectPoseEstimator(
                        objects=self.mesh_files, K=self.K,
                        weights_dir=self.weights_dir,
                        est_refine_iter=self.est_refine_iter,
                        track_refine_iter=self.track_refine_iter,
                        debug=0, debug_dir=self.debug_dir,
                    )
                print("[FoundationPose] estimate_all()：首帧多目标初始化 …")
                valid_masks = filter_valid_masks(masks)
                skipped = set(masks.keys()) - set(valid_masks.keys())
                if skipped:
                    print(f"  [跳过] 掩码像素数不足（可能被遮挡/暂不可见），本帧不估计: {sorted(skipped)}")
                self._infer_result = self.estimator.estimate_all(rgb, valid_masks, depth=depth_m)
            except Exception as e:  # noqa: BLE001 - 后台线程需要把异常带回主线程打印
                print(f"[FoundationPose] estimate_all() 失败: {e}")
                self._infer_error = e
                self._infer_result = {}
            finally:
                self._infer_busy = False

        self._infer_thread = threading.Thread(target=worker, daemon=True)
        self._infer_thread.start()

    def poll_async_result(self):
        """在后台估计线程已完成（is_busy 刚变为 False）后调用一次，取出
        本轮 estimate_all() 的结果并合并进 latest_poses，同时清空线程句柄。
        返回 True 表示确实有新结果被应用（调用方可据此判断是否需要退出
        "刚进入 tracking 状态、等待首次估计"这一过渡态）。"""
        if self._infer_thread is None:
            return False
        self.apply_poses(self._infer_result or {})
        self._infer_thread = None
        return True

    @property
    def has_pending_async(self):
        return self._infer_thread is not None

    # ── 连续帧增量跟踪（track_all，不需要掩码）────────────────────────────
    def track(self, rgb, depth_m):
        poses = self.estimator.track_all(rgb, depth=depth_m)
        self.apply_poses(poses)
        return poses

    def apply_poses(self, poses):
        """对一批新估计/跟踪出的位姿做合理性过滤后合并进 self.latest_poses。"""
        for name, T_nut_cam in list(poses.items()):
            if not pose_is_plausible(T_nut_cam):
                print(f"  [异常位姿丢弃] '{name}' 深度={T_nut_cam[2, 3]:.3f}m 超出合理范围 "
                      f"{DEPTH_SANITY_RANGE_M}，已重置该目标跟踪状态。")
                del poses[name]
                self.reset(name)
        self.latest_poses.update(poses)

    def visualize_all(self, rgb, colors=None):
        if self.estimator is not None and self.latest_poses:
            return self.estimator.visualize_all(rgb, self.latest_poses, colors=colors)
        import cv2
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ── 坐标变换：相机系螺母位姿 -> 基坐标系抓取/法兰目标位姿 ────────────
    def compute_T_grasp_base(self, name):
        """对已跟踪到位姿的螺母 name，计算：
          1) T_nut_base_est = T_cam_base @ T_nut_cam（换算到机械臂基坐标系）；
          2) 修正薄片对称螺母的 180° 上下翻转姿态歧义；
          3) T_grasp_base_est = T_nut_base_est @ T_grasp_nut（复合抓取偏移，
             得到夹爪 TCP 在基坐标系下的最终 6D 位姿）。
        返回 (T_grasp_base_est, T_nut_base_est)。"""
        T_nut_cam = self.latest_poses[name]
        T_nut_base_est = self.T_cam_base @ T_nut_cam
        T_nut_base_est = fix_flat_object_updown_ambiguity(T_nut_base_est)

        grasp_file = self.grasp_pose_files[name]
        if not os.path.isabs(grasp_file):
            grasp_file = os.path.join(self.here, grasp_file)
        T_grasp_nut_candidates = load_grasp_pose(grasp_file)
        T_grasp_nut = T_grasp_nut_candidates[0]
        T_grasp_base_est = T_nut_base_est @ T_grasp_nut
        return T_grasp_base_est, T_nut_base_est

    def compute_T_flange_base(self, name):
        """在 compute_T_grasp_base() 基础上进一步换算末端法兰的目标位姿：
        T_flange_base_est = T_grasp_base_est @ inv(T_tcp_flange)。"""
        T_grasp_base_est, _ = self.compute_T_grasp_base(name)
        return T_grasp_base_est @ np.linalg.inv(self.T_tcp_flange)

    # ── 诊断：与仿真真值比较，评估感知精度（pos_err/rot_err）────────────
    def compute_pose_error_vs_ground_truth(self, model, data, name):
        """返回 (T_nut_base_est, pos_err_m, rot_err_deg)，仅用于调试/精度
        评估——直接读取仿真中螺母 body 的真实位姿作为 ground truth，与感知
        估计出的 T_nut_base_est 比较。真实机器人上没有 ground truth，此
        方法仅在 MuJoCo 仿真环境下有意义。"""
        _, T_nut_base_est = self.compute_T_grasp_base(name)
        T_nut_world_gt = get_body_T_world(model, data, name)
        T_nut_base_gt = np.linalg.inv(self.T_base_world) @ T_nut_world_gt
        pos_err = np.linalg.norm(T_nut_base_est[:3, 3] - T_nut_base_gt[:3, 3])
        rot_err_deg = np.degrees(np.linalg.norm(
            rotation_log(T_nut_base_est[:3, :3] @ T_nut_base_gt[:3, :3].T)))
        return T_nut_base_est, pos_err, rot_err_deg
