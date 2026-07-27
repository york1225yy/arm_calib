#!/usr/bin/env python3
"""
FoundationPose 单文件调用接口
=====================================
支持视频文件 / 摄像头实时 6D 姿态估计与跟踪。
将本文件连同 weights/ 目录一起复制到任意项目即可使用。

目录结构（放置后）:
  your_project/
  ├── foundationpose_api.py       ← 本文件
  ├── weights/                    ← 预训练权重（见 download_assets.py）
  │   ├── 2023-10-28-18-33-37/
  │   └── 2024-01-11-20-02-45/
  └── FoundationPose/             ← 源码（本文件上一级即可）

使用示例:
  # 摄像头实时推理（框选目标后开始）
  python foundationpose_api.py --mesh demo_data/mustard0/mesh/textured_simple.obj --input 0

  # 视频文件推理
  python foundationpose_api.py --mesh demo_data/mustard0/mesh/textured_simple.obj \\
      --input demo_data/mustard0 --depth_dir demo_data/mustard0

  # 指定相机内参（fx fy cx cy）
  python foundationpose_api.py --mesh obj.obj --input 0 --K "600 600 320 240"

  # 多目标（同一帧内多个物体，如场景中的两个螺母）：
  # 掩码目录按目标名分子目录（masks/square_nut/、masks/square_nut_2/，见
  # 05_collect_pose_estimation_data_mujoco.py 的采集格式）时自动进入多目标模式，
  # 所有目标默认共用同一个 --mesh，可用 --object_mesh NAME=PATH 单独覆盖：
  python foundationpose_api.py --mesh nut_mesh/textured_simple.obj \\
      --input pose_estimation_data --cam_K_file pose_estimation_data/cam_K.txt \\
      --save_vis
  # 每个目标的姿态矩阵分别保存至 output/poses/<目标名>/*.txt，
  # 可视化结果（所有目标叠加在同一张图上）保存至 output/vis/*.png。
  # 代码中直接调用见 MultiObjectPoseEstimator（本文件内）。

操作说明:
  - 第一帧会弹出窗口，用鼠标拖动框选目标物体，按 Enter/Space 确认
  - 运行时按 q 退出，按 r 重新框选（重新初始化姿态）
"""

import argparse
import logging
import os
import sys
import glob
import time
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────
#  定位 FoundationPose 源码根目录
#  查找顺序：
#    1. 环境变量 FOUNDATIONPOSE_ROOT
#    2. 本文件所在目录的上一级（standalone/ 放在项目内时）
#    3. 本文件所在目录本身（直接复制到项目根时）
# ─────────────────────────────────────────────
# 若未手动设置环境变量，则使用此默认路径（不会覆盖已有的环境变量）
os.environ.setdefault("FOUNDATIONPOSE_ROOT", "/home/byd/ws/foundationpose")


def _find_fp_root() -> str:
    env = os.environ.get("FOUNDATIONPOSE_ROOT", "")
    if env and os.path.isfile(os.path.join(env, "estimater.py")):
        return env

    this_dir = Path(__file__).resolve().parent
    for candidate in [this_dir.parent, this_dir]:
        if (candidate / "estimater.py").exists():
            return str(candidate)

    raise RuntimeError(
        "找不到 FoundationPose 源码根目录。\n"
        "请设置环境变量 FOUNDATIONPOSE_ROOT=<路径> 或将本文件放在源码目录下。"
    )


FP_ROOT = _find_fp_root()
if FP_ROOT not in sys.path:
    sys.path.insert(0, FP_ROOT)

# ─────────────────────────────────────────────
#  延迟导入（需要 CUDA + 编译好的扩展）
# ─────────────────────────────────────────────
def _import_fp():
    from estimater import (
        FoundationPose,
        ScorePredictor,
        PoseRefinePredictor,
    )
    from Utils import (
        set_logging_format,
        set_seed,
        draw_posed_3d_box,
        draw_xyz_axis,
    )
    import trimesh
    import nvdiffrast.torch as dr

    return FoundationPose, ScorePredictor, PoseRefinePredictor, \
           set_logging_format, set_seed, draw_posed_3d_box, draw_xyz_axis, \
           trimesh, dr


# ─────────────────────────────────────────────
#  辅助：从目录或视频读取帧
# ─────────────────────────────────────────────
class FrameSource:
    """统一的帧源抽象，支持：目录序列 / 视频文件 / 摄像头。"""

    def __init__(self, rgb_source: str, depth_source: str | None = None):
        self.depth_source = depth_source
        self._depth_dir: str | None = None  # 深度目录（按文件名匹配）
        self._cap_depth: cv2.VideoCapture | None = None

        src = str(rgb_source)

        # ── RGB 源 ──
        if os.path.isdir(src):
            # 目录：读取 rgb/ 子目录的 png/jpg
            candidates = sorted(glob.glob(os.path.join(src, "rgb", "*.png")))
            if not candidates:
                candidates = sorted(glob.glob(os.path.join(src, "rgb", "*.jpg")))
            if not candidates:
                # 目录下直接存放图片
                candidates = sorted(
                    glob.glob(os.path.join(src, "*.png"))
                    + glob.glob(os.path.join(src, "*.jpg"))
                )
            self._rgb_files = candidates
            self._mode = "dir"
            self._idx = 0

            # 深度目录（按文件名匹配，与 datareader 一致）
            if depth_source is None:
                depth_dir = os.path.join(src, "depth")
                if os.path.isdir(depth_dir):
                    self._depth_dir = depth_dir
            else:
                if os.path.isdir(depth_source):
                    # 优先查找 depth_source/depth/ 子目录
                    sub = os.path.join(depth_source, "depth")
                    if os.path.isdir(sub):
                        self._depth_dir = sub
                    else:
                        self._depth_dir = depth_source

        elif src.isdigit() or (len(src) == 1 and src in "0123456789"):
            # 摄像头
            self._cap = cv2.VideoCapture(int(src))
            self._mode = "cam"
            if depth_source and os.path.isfile(depth_source):
                self._cap_depth = cv2.VideoCapture(depth_source)
        else:
            # 视频文件
            self._cap = cv2.VideoCapture(src)
            self._mode = "video"
            if depth_source and os.path.isfile(depth_source):
                self._cap_depth = cv2.VideoCapture(depth_source)

    def read(self):
        """返回 (rgb_uint8_HxWx3, depth_float32_HxW_or_None)"""
        # RGB
        if self._mode == "dir":
            if self._idx >= len(self._rgb_files):
                return None, None
            rgb_path = self._rgb_files[self._idx]
            rgb = cv2.imread(rgb_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # BGR→RGB（连续内存）
            depth = None
            # 按文件名匹配深度（与 datareader 一致）
            if self._depth_dir is not None:
                depth_path = os.path.join(
                    self._depth_dir, os.path.basename(rgb_path)
                )
                if os.path.isfile(depth_path):
                    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(
                        np.float32
                    ) / 1e3  # mm→m
                    depth[(depth < 0.001) | (depth >= np.inf)] = 0
            self._idx += 1
            return rgb, depth
        else:
            ret, frame = self._cap.read()
            if not ret:
                return None, None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth = None
            if self._cap_depth is not None:
                ret_d, frame_d = self._cap_depth.read()
                if ret_d:
                    depth = cv2.cvtColor(frame_d, cv2.COLOR_BGR2GRAY).astype(
                        np.float32
                    ) / 1e3
            return rgb, depth

    def release(self):
        if self._mode != "dir":
            self._cap.release()
        if self._cap_depth is not None:
            self._cap_depth.release()

    @property
    def is_finite(self):
        return self._mode in ("dir", "video")


# ─────────────────────────────────────────────
#  辅助：从摄像头内参字符串解析 K 矩阵
# ─────────────────────────────────────────────
def parse_K(k_str: str | None, H: int, W: int) -> np.ndarray:
    if k_str:
        vals = list(map(float, k_str.split()))
        if len(vals) == 4:
            fx, fy, cx, cy = vals
        elif len(vals) == 9:
            return np.array(vals, dtype=float).reshape(3, 3)
        else:
            raise ValueError("--K 需要 4 个值（fx fy cx cy）或 9 个值（行优先 3x3）")
    else:
        # 根据图像尺寸估算（适用于一般广角摄像头）
        fx = fy = max(W, H) * 1.2
        cx, cy = W / 2.0, H / 2.0
        logging.warning(
            f"未指定相机内参，使用估算值 fx=fy={fx:.1f}, cx={cx:.1f}, cy={cy:.1f}。"
            "如需精确结果，请用 --K 'fx fy cx cy' 传入真实标定参数。"
        )
    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


# ─────────────────────────────────────────────
#  辅助：用鼠标框选 ROI → 二值掩码
# ─────────────────────────────────────────────
def select_mask_by_roi(rgb: np.ndarray, window_name: str = "框选目标 [Enter/Space确认, C取消]") -> np.ndarray | None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    roi = cv2.selectROI(window_name, bgr, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    x, y, w, h = roi
    if w == 0 or h == 0:
        return None
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask[y: y + h, x: x + w] = 255
    return mask


# ─────────────────────────────────────────────
#  辅助：合成深度（无深度摄像头时使用）
# ─────────────────────────────────────────────
def make_synthetic_depth(H: int, W: int, assumed_z: float = 0.5) -> np.ndarray:
    """生成全图为固定距离的假深度图（单位：m）。
    注意：无真实深度时姿态中的 z 平移分量不准确，旋转估计仍有效。
    """
    return np.full((H, W), assumed_z, dtype=np.float32)


# ─────────────────────────────────────────────
#  主类：FoundationPoseTracker
# ─────────────────────────────────────────────
class FoundationPoseTracker:
    """
    封装好的 FoundationPose 推理接口。

    参数
    ----
    mesh_file : str
        物体 CAD 模型路径（.obj / .ply / .stl 等 trimesh 支持的格式）
    weights_dir : str
        权重目录，下面包含 refiner 和 scorer 两个子文件夹
    K : np.ndarray (3,3)
        相机内参矩阵
    est_refine_iter : int
        初始估计的精化迭代次数（越大越准但越慢，默认 5）
    track_refine_iter : int
        跟踪阶段的精化迭代次数（默认 2）
    assumed_depth : float
        无深度图时使用的假设物体距离（单位：m，默认 0.5）
    debug : int
        调试级别（0=关闭，1=显示，2=保存中间文件）
    debug_dir : str
        调试结果保存目录
    """

    def __init__(
        self,
        mesh_file: str,
        weights_dir: str,
        K: np.ndarray,
        est_refine_iter: int = 5,
        track_refine_iter: int = 2,
        assumed_depth: float = 0.5,
        debug: int = 1,
        debug_dir: str = "output",
        scorer=None,
        refiner=None,
        glctx=None,
    ):
        """
        scorer / refiner / glctx : 可选，外部已创建好的共享资源（见
            `FoundationPoseTracker.create_shared_resources`）。用于
            `MultiObjectPoseEstimator` 在同一进程内同时估计多个目标时，
            避免每个目标都重复加载一遍权重（ScorePredictor/PoseRefinePredictor）
            和创建 CUDA 光栅化上下文（RasterizeCudaContext），显著节省显存和
            初始化耗时。三者要么都不提供（内部各自创建，行为与之前完全一致），
            要么都提供（生命周期由外部管理）。
        """
        (
            FoundationPose, ScorePredictor, PoseRefinePredictor,
            set_logging_format, set_seed,
            draw_posed_3d_box, draw_xyz_axis,
            trimesh, dr,
        ) = _import_fp()

        set_logging_format()
        set_seed(0)

        self._draw_box = draw_posed_3d_box
        self._draw_axis = draw_xyz_axis
        self._est_refine_iter = est_refine_iter
        self._track_refine_iter = track_refine_iter
        self._assumed_depth = assumed_depth
        self._K = K
        self.debug = debug
        self.debug_dir = debug_dir
        os.makedirs(debug_dir, exist_ok=True)

        shared = scorer is not None or refiner is not None or glctx is not None
        if shared and not (scorer is not None and refiner is not None and glctx is not None):
            raise ValueError("scorer/refiner/glctx 要么都提供，要么都不提供。")

        if shared:
            self._scorer = scorer
            self._refiner = refiner
            self._glctx = glctx
        else:
            # ── 加载权重 ──
            # 自动寻找 refiner / scorer 子目录
            if not os.path.isdir(weights_dir):
                raise FileNotFoundError(f"权重目录不存在: {weights_dir}")

            os.environ["FOUNDATIONPOSE_WEIGHTS_DIR"] = weights_dir

            logging.info("正在加载 ScorePredictor …")
            self._scorer = ScorePredictor()
            logging.info("正在加载 PoseRefinePredictor …")
            self._refiner = PoseRefinePredictor()

            self._glctx = dr.RasterizeCudaContext()

        # ── 加载网格 ──
        logging.info(f"正在加载网格: {mesh_file}")
        self._mesh = trimesh.load(mesh_file)
        to_origin, extents = trimesh.bounds.oriented_bounds(self._mesh)
        self._bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        self._to_origin = to_origin

        # ── 初始化估计器 ──
        self._est = FoundationPose(
            model_pts=self._mesh.vertices,
            model_normals=self._mesh.vertex_normals,
            mesh=self._mesh,
            scorer=self._scorer,
            refiner=self._refiner,
            debug_dir=debug_dir,
            debug=debug,
            glctx=self._glctx,
        )

        self._pose = None  # 当前姿态（已初始化则非 None）
        logging.info("FoundationPoseTracker 初始化完成")

    # ─────────────────────────────────────────
    @staticmethod
    def create_shared_resources(weights_dir: str):
        """创建可在多个 FoundationPoseTracker 实例间共享的重资源
        （ScorePredictor / PoseRefinePredictor / RasterizeCudaContext）。

        供 `MultiObjectPoseEstimator` 内部使用：同一帧里的多个目标（如场景中
        的多个螺母）通常复用同一套权重和 CUDA 上下文，没必要每个目标各自
        重新加载一遍，可显著节省显存占用和初始化耗时。

        返回
        ----
        (scorer, refiner, glctx) 三元组，可直接传给
        `FoundationPoseTracker(..., scorer=, refiner=, glctx=)`。
        """
        (
            _FoundationPose, ScorePredictor, PoseRefinePredictor,
            set_logging_format, set_seed,
            _draw_posed_3d_box, _draw_xyz_axis,
            _trimesh, dr,
        ) = _import_fp()

        set_logging_format()
        set_seed(0)

        if not os.path.isdir(weights_dir):
            raise FileNotFoundError(f"权重目录不存在: {weights_dir}")
        os.environ["FOUNDATIONPOSE_WEIGHTS_DIR"] = weights_dir

        logging.info("正在加载共享 ScorePredictor …")
        scorer = ScorePredictor()
        logging.info("正在加载共享 PoseRefinePredictor …")
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        return scorer, refiner, glctx

    # ─────────────────────────────────────────
    def reset(self):
        """重置跟踪状态（下一帧将重新执行姿态估计）。"""
        self._pose = None
        self._est.pose_last = None

    # ─────────────────────────────────────────
    def initialize(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        depth: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        第一帧初始化：执行姿态估计。

        参数
        ----
        rgb   : uint8 HxWx3（RGB 顺序）
        mask  : uint8 HxW，255=目标区域，0=背景
        depth : float32 HxW，单位 m；None 则使用假设深度

        返回
        ----
        pose : float64 4x4，物体坐标系在相机坐标系下的变换矩阵
        """
        if depth is None:
            logging.warning("未提供深度图，使用估算深度（姿态 z 分量不准确）。")
            depth = make_synthetic_depth(*rgb.shape[:2], self._assumed_depth)

        pose = self._est.register(
            K=self._K,
            rgb=rgb,
            depth=depth,
            ob_mask=mask.astype(bool),
            iteration=self._est_refine_iter,
        )
        self._pose = pose
        return pose

    # ─────────────────────────────────────────
    def track(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        后续帧跟踪：调用前须先调用 initialize()。

        参数
        ----
        rgb   : uint8 HxWx3
        depth : float32 HxW，单位 m；None 则使用假设深度

        返回
        ----
        pose : float64 4x4
        """
        if self._pose is None:
            raise RuntimeError("请先调用 initialize() 初始化姿态。")
        if depth is None:
            depth = make_synthetic_depth(*rgb.shape[:2], self._assumed_depth)

        pose = self._est.track_one(
            rgb=rgb,
            depth=depth,
            K=self._K,
            iteration=self._track_refine_iter,
        )
        self._pose = pose
        return pose

    # ─────────────────────────────────────────
    def visualize(self, rgb: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """
        在图像上绘制 3D 包围框和坐标轴。

        参数
        ----
        rgb  : uint8 HxWx3（RGB 顺序）
        pose : 4x4 姿态矩阵

        返回
        ----
        vis : uint8 HxWx3 BGR，可直接传给 cv2.imshow
        """
        center_pose = pose @ np.linalg.inv(self._to_origin)
        vis = self._draw_box(
            self._K, img=rgb.copy(), ob_in_cam=center_pose, bbox=self._bbox
        )
        vis = self._draw_axis(
            vis,
            ob_in_cam=center_pose,
            scale=0.1,
            K=self._K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        # 返回 BGR 便于 cv2.imshow
        return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────
#  独立功能 API：6DoF 位姿估计
#  （供其他脚本 / 模块直接 import 调用，无需关心 CLI 参数）
#  功能与 FoundationPoseTracker（原 CLI 主循环使用的类）完全一致：
#    - estimate() ≙ 原 initialize()：首帧 / 需要重新初始化时调用，依赖 mask，
#      内部走 register 流程（旋转假设采样 + 迭代精化 + 打分选优），较慢但准。
#    - track()    ≙ 原 track()     ：后续帧调用，不需要 mask，依赖上一帧姿态
#      做增量跟踪（track_one），更快，适合摄像头/视频的连续帧场景。
# ─────────────────────────────────────────────
class PoseEstimatorAPI:
    """
    6DoF 位姿估计的简洁封装，供其他功能模块直接调用，功能与原
    FoundationPoseTracker 完全一致（对摄像头/视频连续帧同样适用）。

    典型用法（针对摄像头/视频连续帧）
    ----
        import numpy as np
        from foundationpose_api import PoseEstimatorAPI

        K = np.loadtxt("pose_estimation_data/cam_K.txt").reshape(3, 3)
        estimator = PoseEstimatorAPI(mesh_file="nut_mesh/textured.obj", K=K)

        # 第一帧（或需要重新初始化时）：提供 mask，执行完整估计
        pose_4x4 = estimator.estimate(rgb=rgb_img, mask=mask_img, depth=depth_img)

        # 后续帧：不需要 mask，基于上一帧姿态做增量跟踪，速度更快
        while ...:
            pose_4x4 = estimator.track(rgb=rgb_img, depth=depth_img)

        # pose_4x4: np.ndarray, shape (4, 4), float64
        #   物体坐标系 -> 相机坐标系（OpenCV 约定：+X右 +Y下 +Z前）的齐次变换矩阵

    单帧独立估计（不做连续跟踪）的用法
    ----
        pose_4x4 = estimator.estimate(rgb=rgb_img, mask=mask_img, depth=depth_img, reinit=True)
        # reinit=True（默认）：调用前先清空历史跟踪状态，保证与之前的调用互不影响
    """

    def __init__(
        self,
        mesh_file: str,
        K: np.ndarray,
        weights_dir: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "weights"
        ),
        est_refine_iter: int = 5,
        track_refine_iter: int = 2,
        debug: int = 0,
        debug_dir: str = "output",
    ):
        """
        参数
        ----
        mesh_file : str
            目标物体 CAD 网格路径（.obj/.ply/.stl 等 trimesh 支持的格式）
        K : np.ndarray (3, 3)
            相机内参矩阵
        weights_dir : str
            预训练权重目录（默认：本文件同级的 weights/）
        est_refine_iter : int
            estimate() 的精化迭代次数（越大越准但越慢，默认 5）
        track_refine_iter : int
            track() 的精化迭代次数（默认 2，跟踪阶段通常帧间运动小，迭代次数可以更少）
        debug : int
            调试级别（0=关闭，1=显示，2=保存中间文件）
        debug_dir : str
            调试结果保存目录
        """
        self._tracker = FoundationPoseTracker(
            mesh_file=mesh_file,
            weights_dir=weights_dir,
            K=K,
            est_refine_iter=est_refine_iter,
            track_refine_iter=track_refine_iter,
            debug=debug,
            debug_dir=debug_dir,
        )

    @property
    def is_initialized(self) -> bool:
        """是否已经完成过至少一次 estimate()，可以开始调用 track()。"""
        return self._tracker._pose is not None

    def estimate(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        depth: np.ndarray | None = None,
        reinit: bool = True,
    ) -> np.ndarray:
        """
        执行一次完整的姿态估计（对应原 FoundationPoseTracker.initialize()）。

        用于：视频/摄像头的第一帧初始化，或跟踪丢失/需要重新初始化时调用。

        参数
        ----
        rgb   : uint8, (H, W, 3)
            RGB 通道顺序（非 BGR）
        mask  : uint8 / bool, (H, W)
            目标物体二值掩码（>0 或 True 表示目标区域）
        depth : float32, (H, W)，单位：米，可选
            不提供时使用假设深度（旋转估计仍有效，但 z 方向平移不准确）
        reinit : bool
            True（默认）：调用前先清空内部跟踪状态，保证本次是与历史完全无关的
            独立估计，适合单帧场景反复调用。
            False：不清空，估计结果会正常更新跟踪状态，可紧接着调用 track()
            继续对后续帧做增量跟踪（等价于原代码里首帧调用 initialize()）。

        返回
        ----
        pose : np.ndarray, float64, (4, 4)
            物体坐标系 -> 相机坐标系 的齐次变换矩阵
        """
        if reinit:
            self._tracker.reset()
        return self._tracker.initialize(rgb=rgb, mask=mask, depth=depth)

    def track(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        对后续帧执行增量跟踪（对应原 FoundationPoseTracker.track()）。

        不需要 mask，复用上一帧姿态作为起点做精化，速度比 estimate() 快，
        适合摄像头/视频连续帧场景。调用前必须已通过 estimate() 完成过一次
        初始化，否则会抛出 RuntimeError。

        参数
        ----
        rgb   : uint8, (H, W, 3)
        depth : float32, (H, W)，单位：米，可选

        返回
        ----
        pose : np.ndarray, float64, (4, 4)
        """
        return self._tracker.track(rgb=rgb, depth=depth)

    def reset(self) -> None:
        """清空跟踪状态。下一次 estimate() 会视为全新的独立估计。"""
        self._tracker.reset()

    def visualize(self, rgb: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """在图像上绘制 3D 包围框与坐标轴，返回 BGR 图像，便于调试可视化。"""
        return self._tracker.visualize(rgb, pose)


def estimate_6dof_pose(
    mesh_file: str,
    rgb: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    depth: np.ndarray | None = None,
    weights_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "weights"
    ),
    est_refine_iter: int = 5,
    debug: int = 0,
    debug_dir: str = "output",
) -> np.ndarray:
    """
    单次调用的便捷函数版本（内部临时创建 PoseEstimatorAPI 并调用一次）。

    注意：权重和网格加载本身较慢（秒级），如果调用方需要连续/多次估计，
    请直接使用 PoseEstimatorAPI 类并复用同一个实例，避免每次调用都重新加载。

    参数、返回值同 PoseEstimatorAPI.estimate()，额外增加 mesh_file/weights_dir/
    est_refine_iter/debug/debug_dir 用于一次性构造 Tracker。
    """
    estimator = PoseEstimatorAPI(
        mesh_file=mesh_file,
        K=K,
        weights_dir=weights_dir,
        est_refine_iter=est_refine_iter,
        debug=debug,
        debug_dir=debug_dir,
    )
    return estimator.estimate(rgb=rgb, mask=mask, depth=depth)


# ─────────────────────────────────────────────
#  多目标位姿估计：同一帧内对场景中的多个目标（如两个螺母）分别估计，
#  并支持把所有目标的可视化结果叠加输出到同一张图上。
# ─────────────────────────────────────────────
class MultiObjectPoseEstimator:
    """
    对同一帧图像中的多个目标物体分别做 6D 姿态估计，并支持在同一张图上
    叠加输出各自的可视化结果（3D 包围框 + 坐标轴 + 名称标签）。

    每个目标物体用一个唯一名称标识（例如 gen3_with_two_nuts.xml 场景里的
    "square_nut" / "square_nut_2"），内部为每个名称维护独立的姿态/跟踪状态
    （互不影响），但共享同一份权重（ScorePredictor / PoseRefinePredictor）
    和 CUDA 光栅化上下文，避免多个目标重复加载权重带来的显存/时间开销。

    典型用法（配合 05_collect_pose_estimation_data_mujoco.py 采集的
    masks/<物体名>/frame_XXXXXX.png 目录结构）
    ----
        import cv2, numpy as np
        from foundationpose_api import MultiObjectPoseEstimator

        K = np.loadtxt("pose_estimation_data/cam_K.txt").reshape(3, 3)
        estimator = MultiObjectPoseEstimator(
            objects={
                "square_nut":   "nut_mesh/textured_simple.obj",
                "square_nut_2": "nut_mesh/textured_simple.obj",
            },
            K=K,
        )

        rgb = ...      # (H, W, 3) uint8, RGB 顺序
        depth = ...    # (H, W) float32, 单位米，可选
        masks = {
            "square_nut":   cv2.imread(".../masks/square_nut/frame_000000.png", cv2.IMREAD_GRAYSCALE),
            "square_nut_2": cv2.imread(".../masks/square_nut_2/frame_000000.png", cv2.IMREAD_GRAYSCALE),
        }

        poses = estimator.estimate_all(rgb, masks, depth=depth)
        # poses: dict[str, np.ndarray(4,4)]，每个物体在相机坐标系下的位姿

        vis_bgr = estimator.visualize_all(rgb, poses)
        cv2.imwrite("output/vis/000000.png", vis_bgr)
    """

    def __init__(
        self,
        objects: dict[str, str],
        K: np.ndarray,
        weights_dir: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "weights"
        ),
        est_refine_iter: int = 5,
        track_refine_iter: int = 2,
        debug: int = 0,
        debug_dir: str = "output",
    ):
        """
        参数
        ----
        objects : dict[str, str]
            {目标名: mesh 文件路径}。多个目标名可以指向同一个 mesh 文件
            （如场景中两个外形相同、姿态不同的螺母），mesh 会各自独立加载
            一份（trimesh 加载本身很快），但权重/CUDA 上下文只加载一次。
        K, weights_dir, est_refine_iter, track_refine_iter, debug, debug_dir :
            含义同 `PoseEstimatorAPI`，作用于内部所有目标共用。
        """
        if not objects:
            raise ValueError("objects 不能为空，至少需要指定一个目标名及其 mesh 路径。")

        self._scorer, self._refiner, self._glctx = \
            FoundationPoseTracker.create_shared_resources(weights_dir)

        self._trackers: dict[str, FoundationPoseTracker] = {}
        for name, mesh_file in objects.items():
            logging.info(f"[MultiObjectPoseEstimator] 初始化目标 '{name}' ← {mesh_file}")
            self._trackers[name] = FoundationPoseTracker(
                mesh_file=mesh_file,
                weights_dir=weights_dir,
                K=K,
                est_refine_iter=est_refine_iter,
                track_refine_iter=track_refine_iter,
                debug=debug,
                debug_dir=debug_dir,
                scorer=self._scorer,
                refiner=self._refiner,
                glctx=self._glctx,
            )

    @property
    def object_names(self) -> list[str]:
        """所有已注册的目标名称列表。"""
        return list(self._trackers.keys())

    def is_initialized(self, name: str) -> bool:
        """指定目标是否已完成过至少一次 estimate_all()（可以开始 track_all()）。"""
        return self._trackers[name]._pose is not None

    def reset(self, name: str | None = None) -> None:
        """重置指定目标（或全部目标，name=None）的跟踪状态。"""
        names = [name] if name is not None else list(self._trackers.keys())
        for n in names:
            self._trackers[n].reset()

    def estimate_all(
        self,
        rgb: np.ndarray,
        masks: dict[str, np.ndarray | None],
        depth: np.ndarray | None = None,
        reinit: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        对 `masks` 中提供了有效掩码的目标分别执行一次完整姿态估计
        （对应 `PoseEstimatorAPI.estimate()`，逐个目标调用底层 register）。

        参数
        ----
        rgb   : uint8, (H, W, 3)，RGB 顺序，同一帧内所有目标共用
        masks : dict[名称, mask 或 None]
            每个目标各自的二值掩码；名称不在 `self.object_names` 中的条目
            会被忽略并打印警告；mask 为 None 或全 0（该目标当前不可见）的
            条目会被跳过（不会报错，也不会写入返回结果）。
        depth : float32, (H, W)，可选，同一帧内所有目标共用
        reinit : 同 `PoseEstimatorAPI.estimate()`：True（默认）时每个目标
            调用前都先清空该目标自己的跟踪历史，视为独立估计。

        返回
        ----
        dict[名称, pose_4x4]，只包含本次成功执行了估计的目标。
        """
        poses = {}
        for name, mask in masks.items():
            if name not in self._trackers:
                logging.warning(f"[MultiObjectPoseEstimator] 未知目标名 '{name}'，跳过。")
                continue
            if mask is None or np.count_nonzero(mask) == 0:
                continue
            tracker = self._trackers[name]
            if reinit:
                tracker.reset()
            poses[name] = tracker.initialize(rgb=rgb, mask=mask, depth=depth)
        return poses

    def track_all(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None = None,
        only_initialized: bool = True,
        exclude: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        对已初始化过的目标分别执行增量跟踪（对应 `PoseEstimatorAPI.track()`）。

        参数
        ----
        rgb, depth : 同 `estimate_all`
        only_initialized : True（默认）时静默跳过尚未初始化的目标；
            False 时遇到未初始化的目标会抛出 RuntimeError。
        exclude : 可选，本次调用要跳过的目标名集合（典型用途：同一帧里刚
            通过 `estimate_all()` 重新初始化过的目标，不需要在同一帧里
            紧接着再 track 一次）。

        返回
        ----
        dict[名称, pose_4x4]
        """
        exclude = exclude or set()
        poses = {}
        for name, tracker in self._trackers.items():
            if name in exclude:
                continue
            if tracker._pose is None:
                if only_initialized:
                    continue
                raise RuntimeError(f"目标 '{name}' 尚未初始化，请先调用 estimate_all()。")
            poses[name] = tracker.track(rgb=rgb, depth=depth)
        return poses

    def visualize_all(
        self,
        rgb: np.ndarray,
        poses: dict[str, np.ndarray],
        colors: dict[str, tuple[int, int, int]] | None = None,
    ) -> np.ndarray:
        """
        在同一张图上叠加绘制多个目标各自的 3D 包围框 + 坐标轴 + 名称标签。

        参数
        ----
        rgb    : uint8, (H, W, 3)，RGB 顺序
        poses  : dict[名称, pose_4x4]，通常是 `estimate_all`/`track_all` 的返回值
        colors : 可选，dict[名称, (B,G,R)]，每个目标名称标签文字的颜色
                 （3D 包围框/坐标轴颜色由 FoundationPose 内部固定，不受此参数
                 影响），不提供时按预设调色板循环分配。

        返回
        ----
        vis : uint8 (H, W, 3) BGR，可直接 cv2.imshow / cv2.imwrite
        """
        palette = [(0, 255, 255), (255, 0, 255), (255, 255, 0),
                   (0, 165, 255), (255, 128, 0), (128, 0, 255)]
        vis = None
        for i, (name, pose) in enumerate(poses.items()):
            tracker = self._trackers.get(name)
            if tracker is None:
                continue
            base_rgb = rgb if vis is None else cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            vis = tracker.visualize(base_rgb, pose)

            color = (colors or {}).get(name, palette[i % len(palette)])
            center_cam = pose[:3, 3]
            if center_cam[2] > 1e-6:
                uv = tracker._K @ center_cam
                u, v = int(uv[0] / uv[2]), int(uv[1] / uv[2])
                cv2.putText(vis, name, (u + 6, v - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        if vis is None:
            # 没有任何有效姿态时，原样返回（转 BGR）以便调用方仍可显示/保存
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return vis


# ─────────────────────────────────────────────
#  多目标 CLI 辅助函数
# ─────────────────────────────────────────────
def _discover_multi_object_masks(mask_root: str | None) -> dict[str, str] | None:
    """检测 mask_root 是否为『每个目标一个子目录』的多目标掩码结构（如
    05_collect_pose_estimation_data_mujoco.py 采集的 masks/square_nut/、
    masks/square_nut_2/），是则返回 {目标名: 该目标掩码子目录路径}；不是
    （目录下直接是 *.png，或目录不存在）则返回 None，调用方应退回单目标模式。
    """
    if not mask_root or not os.path.isdir(mask_root):
        return None
    result = {}
    for entry in sorted(os.listdir(mask_root)):
        sub = os.path.join(mask_root, entry)
        if os.path.isdir(sub) and glob.glob(os.path.join(sub, "*.png")):
            result[entry] = sub
    return result or None


def _parse_object_mesh_overrides(entries: list[str]) -> dict[str, str]:
    """解析 --object_mesh 'NAME=PATH' 参数列表为 {名称: mesh 路径} 字典。"""
    overrides = {}
    for item in entries:
        if "=" not in item:
            raise ValueError(f"--object_mesh 格式错误（应为 NAME=PATH）: {item}")
        name, path = item.split("=", 1)
        overrides[name.strip()] = path.strip()
    return overrides


def _run_multi_object_cli(args, source: "FrameSource", K: np.ndarray,
                          mask_dirs: dict[str, str], rgb0: np.ndarray,
                          depth0: np.ndarray | None) -> None:
    """多目标模式主循环：对每一帧同时估计 mask_dirs 中所有目标的 6D 姿态，
    并把所有目标的可视化结果叠加输出到同一张图上。

    与单目标模式的主循环相比：
      - 每个目标独立维护自己的姿态/跟踪状态（MultiObjectPoseEstimator 内部管理）；
      - 每一帧里，某个目标若尚未初始化（或开启 --reinit_each_frame）且当前帧
        存在该目标的有效掩码，则对其执行 estimate（register）；否则（已初始化
        且本帧不重新初始化）对其执行 track；若尚未初始化且当前帧也没有掩码，
        则该目标本帧不产生姿态（等下一帧掩码出现后再初始化）；
      - 保存结果时，每个目标的姿态矩阵分别保存至 output_dir/poses/<目标名>/*.txt，
        可视化结果叠加到同一张图后保存至 output_dir/vis/*.png。
    """
    mesh_overrides = _parse_object_mesh_overrides(args.object_mesh)
    objects = {name: mesh_overrides.get(name, args.mesh) for name in mask_dirs}

    print(f"[多目标模式] 检测到 {len(objects)} 个目标: {list(objects.keys())}")
    for name, mesh_file in objects.items():
        print(f"    {name:<16s} ← {mesh_file}")

    estimator = MultiObjectPoseEstimator(
        objects=objects,
        K=K,
        weights_dir=args.weights_dir,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        debug=args.debug,
        debug_dir=args.output_dir,
    )

    def _load_mask(name, idx):
        files = sorted(glob.glob(os.path.join(mask_dirs[name], "*.png")))
        if idx >= len(files):
            return None
        m = cv2.imread(files[idx], cv2.IMREAD_GRAYSCALE)
        if m is None or m.max() == 0:
            return None
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        return m

    pose_dir = os.path.join(args.output_dir, "poses")
    vis_dir = os.path.join(args.output_dir, "vis")
    if args.save_pose:
        for name in objects:
            os.makedirs(os.path.join(pose_dir, name), exist_ok=True)
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    H, W = rgb0.shape[:2]
    writer = None
    if args.save_video:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 30, (W, H))
        print(f"[信息] 结果视频将保存至 {out_path}")

    if args.reinit_each_frame:
        print("[信息] --reinit_each_frame 已开启：每一帧只要目标存在有效掩码文件，"
              "都会独立重新执行姿态估计（register）；无对应掩码的帧仍走 track() 跟踪。\n")

    frame_idx = 0
    cur_rgb, cur_depth = rgb0, depth0

    while True:
        if cur_rgb is None:
            print("[信息] 输入已读取完毕。")
            break

        masks = {name: _load_mask(name, frame_idx) for name in objects}

        # 本帧需要执行 estimate（register）的目标：尚未初始化，或开启了
        # --reinit_each_frame，且当前帧存在该目标的有效掩码
        to_estimate = {
            name: mask for name, mask in masks.items()
            if mask is not None and (
                not estimator.is_initialized(name) or args.reinit_each_frame
            )
        }

        poses = {}
        if to_estimate:
            poses.update(estimator.estimate_all(cur_rgb, to_estimate, depth=cur_depth,
                                                reinit=False))
        poses.update(estimator.track_all(cur_rgb, depth=cur_depth, only_initialized=True,
                                         exclude=set(to_estimate.keys())))

        if poses:
            if args.save_pose:
                for name, pose in poses.items():
                    np.savetxt(
                        os.path.join(pose_dir, name, f"{frame_idx:06d}.txt"),
                        pose.reshape(4, 4),
                    )

            vis_bgr = estimator.visualize_all(cur_rgb, poses)

            if args.save_vis:
                cv2.imwrite(os.path.join(vis_dir, f"{frame_idx:06d}.png"), vis_bgr)
            if args.show:
                cv2.imshow("FoundationPose (multi-object)", vis_bgr)
            if writer is not None:
                writer.write(vis_bgr)
        elif args.show:
            cv2.imshow("FoundationPose (multi-object)",
                       cv2.cvtColor(cur_rgb, cv2.COLOR_RGB2BGR))

        if frame_idx % 50 == 0:
            print(f"[信息] 已处理 {frame_idx} 帧 …（本帧有效姿态: {list(poses.keys())}）")

        if args.show:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[信息] 用户退出。")
                break
            elif key == ord("r"):
                print("[信息] 重新初始化全部目标 …")
                estimator.reset()

        frame_idx += 1
        cur_rgb, cur_depth = source.read()

    source.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    summary = f"[信息] 共处理 {frame_idx} 帧（多目标模式，{len(objects)} 个目标）"
    if args.save_pose:
        summary += f"，各目标姿态矩阵已分别保存至 {pose_dir}/<目标名>/"
    if args.save_vis:
        summary += f"，叠加可视化图片已保存至 {vis_dir}/"
    print(summary)


# ─────────────────────────────────────────────
#  命令行入口
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="FoundationPose 单文件接口：视频/摄像头 6D 姿态估计",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 必需参数
    parser.add_argument(
        "--mesh", required=True,
        help="物体 CAD 模型文件路径（.obj/.ply/.stl 等）",
    )
    # 输入源
    parser.add_argument(
        "--input", default="0",
        help=(
            "输入源：\n"
            "  0,1,2…  摄像头编号（默认 0）\n"
            "  /path/to/video.mp4  视频文件\n"
            "  /path/to/dir/       包含 rgb/ 子目录的数据目录"
        ),
    )
    parser.add_argument(
        "--depth_dir", default=None,
        help="深度数据目录（含 depth/*.png）或深度视频文件。省略则使用估算深度。",
    )
    parser.add_argument(
        "--mask_dir", default=None,
        help=(
            "掩码目录。提供后跳过 GUI 框选，支持 headless 运行。支持两种结构：\n"
            "  单目标：<mask_dir>/*.png（或 <mask_dir>/masks/*.png）\n"
            "  多目标：<mask_dir>/<目标名>/*.png（每个目标一个子目录，如\n"
            "          05_collect_pose_estimation_data_mujoco.py 采集的\n"
            "          masks/square_nut/、masks/square_nut_2/），检测到该结构\n"
            "          会自动进入多目标模式，对每个子目录分别估计位姿并把\n"
            "          结果叠加可视化到同一张图上。"
        ),
    )
    parser.add_argument(
        "--object_mesh", action="append", default=[],
        metavar="NAME=PATH",
        help=(
            "仅多目标模式下使用：为指定目标单独指定 mesh 文件（可重复传入该参数），\n"
            "格式 'NAME=PATH'，例如 --object_mesh square_nut_2=other_mesh.obj。\n"
            "未通过该参数指定 mesh 的目标，默认使用 --mesh 指定的公共 mesh\n"
            "（适用于本仓库两个螺母共用同一个 mesh 的场景）。"
        ),
    )
    parser.add_argument(
        "--reinit_each_frame", action="store_true",
        help=(
            "每一帧都独立重新执行姿态估计（register），而不是依赖上一帧的跟踪结果（track）。\n"
            "仅在该帧存在有效掩码文件时生效；关闭（默认）或该帧无掩码时，仍使用 track() 跟踪。\n"
            "适用于离散关键帧（帧间运动跳变较大，如键盘控制机械臂逐帧采集）的场景。"
        ),
    )
    parser.add_argument(
        "--cam_K_file", default=None,
        help="相机内参文件（3x3 矩阵，空格分隔）。优先级高于 --K。",
    )
    # 权重
    parser.add_argument(
        "--weights_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"),
        help="预训练权重目录（默认：本文件同级的 weights/）",
    )
    # 相机内参
    parser.add_argument(
        "--K", default=None,
        help="相机内参，格式 'fx fy cx cy'（例：'600 600 320 240'）。省略则自动估算。",
    )
    # 推理参数
    parser.add_argument("--est_refine_iter", type=int, default=5, help="初始估计精化迭代次数")
    parser.add_argument("--track_refine_iter", type=int, default=2, help="跟踪精化迭代次数")
    parser.add_argument(
        "--assumed_depth", type=float, default=0.5,
        help="无深度图时的假设物体距离（单位 m，默认 0.5）",
    )
    # 输出
    parser.add_argument("--output_dir", default="output", help="结果保存目录")
    parser.add_argument(
        "--save_video", action="store_true",
        help="将可视化结果保存为 output_dir/result.mp4",
    )
    parser.add_argument(
        "--save_vis", action="store_true",
        help="将每一帧的可视化图片保存至 output_dir/vis/*.png",
    )
    parser.add_argument(
        "--save_pose", dest="save_pose", action="store_true", default=True,
        help="将每一帧的姿态矩阵保存至 output_dir/poses/*.txt（默认开启）",
    )
    parser.add_argument(
        "--no_save_pose", dest="save_pose", action="store_false",
        help="不保存姿态矩阵",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="运行时弹出窗口实时显示可视化结果（需要有显示器）",
    )
    parser.add_argument("--debug", type=int, default=1, help="调试级别（0/1/2）")
    args = parser.parse_args()

    # ── 打开输入源（先读一帧获取分辨率）──
    source = FrameSource(args.input, args.depth_dir)
    rgb0, depth0 = source.read()
    if rgb0 is None:
        print("[错误] 无法读取输入，请检查 --input 参数。")
        sys.exit(1)

    H, W = rgb0.shape[:2]

    # ── 相机内参 ──
    if args.cam_K_file and os.path.isfile(args.cam_K_file):
        K = np.loadtxt(args.cam_K_file).reshape(3, 3)
        logging.info(f"从文件加载相机内参: {args.cam_K_file}")
    else:
        K = parse_K(args.K, H, W)

    # ── 多目标模式检测：掩码目录下是否为『每个目标一个子目录』的结构 ──
    mask_root_candidate = args.mask_dir
    if mask_root_candidate is None and os.path.isdir(str(args.input)):
        auto_mask_dir = os.path.join(str(args.input), "masks")
        if os.path.isdir(auto_mask_dir):
            mask_root_candidate = auto_mask_dir

    multi_masks = _discover_multi_object_masks(mask_root_candidate)
    if multi_masks:
        _run_multi_object_cli(args, source, K, multi_masks, rgb0, depth0)
        return

    # ── 加载掩码文件列表（headless 模式）──
    mask_files: list[str] = []
    if args.mask_dir:
        mask_search = args.mask_dir
        if os.path.isdir(os.path.join(mask_search, "masks")):
            mask_search = os.path.join(mask_search, "masks")
        mask_files = sorted(glob.glob(os.path.join(mask_search, "*.png")))
        if mask_files:
            print(f"[信息] 发现 {len(mask_files)} 个掩码文件，将以 headless 模式运行。")
        else:
            print(f"[警告] --mask_dir 指定但未找到掩码文件: {mask_search}")
    # 也自动检测 input 目录下的 masks/
    elif os.path.isdir(str(args.input)):
        auto_mask_dir = os.path.join(str(args.input), "masks")
        if os.path.isdir(auto_mask_dir):
            mask_files = sorted(glob.glob(os.path.join(auto_mask_dir, "*.png")))
            if mask_files:
                print(f"[信息] 自动检测到 {len(mask_files)} 个掩码文件，以 headless 模式运行。")

    use_mask_files = len(mask_files) > 0
    # --show 强制开启显示；无掩码文件时默认开启（需要 GUI 框选）
    show_gui = args.show or (not use_mask_files)

    if args.reinit_each_frame and not use_mask_files:
        print("[警告] --reinit_each_frame 需要每帧掩码文件（--mask_dir 或 input/masks/），"
              "当前未检测到掩码，将退化为普通跟踪模式。")

    # ── 初始化 Tracker ──
    tracker = FoundationPoseTracker(
        mesh_file=args.mesh,
        weights_dir=args.weights_dir,
        K=K,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        assumed_depth=args.assumed_depth,
        debug=args.debug,
        debug_dir=args.output_dir,
    )

    # ── 视频写入器（可选）──
    writer = None
    if args.save_video:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 30, (W, H))
        print(f"[信息] 结果视频将保存至 {out_path}")

    # ── 主循环 ──
    pose_dir = os.path.join(args.output_dir, "poses")
    vis_dir = os.path.join(args.output_dir, "vis")
    if args.save_pose:
        os.makedirs(pose_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    frame_idx = 0
    initialized = False

    if not use_mask_files:
        print("\n[操作说明]")
        print("  第一帧：用鼠标框选目标物体 → Enter/Space 确认")
        print("  运行时：按 q 退出，按 r 重新初始化\n")
    else:
        if show_gui:
            print("\n[掩码模式 + 实时显示] 使用掩码文件初始化，同时显示可视化窗口。")
            print("  运行时：按 q 退出\n")
        else:
            print("\n[Headless 模式] 使用掩码文件自动初始化，无需 GUI。\n")

    if args.reinit_each_frame and use_mask_files:
        print("[信息] --reinit_each_frame 已开启：每一帧只要存在有效掩码文件，"
              "都会独立重新执行姿态估计（register），不依赖上一帧的跟踪结果；"
              "无对应掩码的帧仍会退回 track() 跟踪。\n")

    def _load_mask_for_frame(idx):
        """尝试为指定帧号加载掩码文件，找不到 / 无效则返回 None。"""
        if not use_mask_files or idx >= len(mask_files):
            return None
        m = cv2.imread(mask_files[idx], cv2.IMREAD_GRAYSCALE)
        if m is None or m.max() == 0:
            return None
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        return m

    # 第一帧直接用已读取的帧
    cur_rgb, cur_depth = rgb0, depth0

    while True:
        if cur_rgb is None:
            print("[信息] 输入已读取完毕。")
            break

        if not initialized:
            # 获取掩码
            if use_mask_files:
                mask = _load_mask_for_frame(frame_idx)
            else:
                mask = select_mask_by_roi(cur_rgb)

            if mask is None:
                print(f"[警告] 第 {frame_idx} 帧无有效掩码，跳过。")
            else:
                print(f"[信息] 第 {frame_idx} 帧：执行姿态估计 …")
                t0 = time.time()
                pose = tracker.initialize(cur_rgb, mask, cur_depth)
                print(f"[信息] 姿态估计耗时 {time.time()-t0:.2f}s")
                initialized = True
        else:
            # 已初始化：默认继续跟踪（track）。若开启 --reinit_each_frame 且当前帧
            # 存在有效掩码文件，则改为对该帧独立重新执行姿态估计（register），
            # 不依赖上一帧的跟踪结果；否则（开关关闭，或该帧没有掩码）仍走 track()。
            reinit_mask = _load_mask_for_frame(frame_idx) if args.reinit_each_frame else None
            if reinit_mask is not None:
                pose = tracker.initialize(cur_rgb, reinit_mask, cur_depth)
            else:
                pose = tracker.track(cur_rgb, cur_depth)

        if initialized:
            # 保存姿态
            if args.save_pose:
                np.savetxt(
                    os.path.join(pose_dir, f"{frame_idx:06d}.txt"),
                    pose.reshape(4, 4),
                )

            # 可视化
            vis_bgr = tracker.visualize(cur_rgb, pose)

            # 保存可视化图片
            if args.save_vis:
                cv2.imwrite(
                    os.path.join(vis_dir, f"{frame_idx:06d}.png"),
                    vis_bgr,
                )

            if show_gui:
                cv2.imshow("FoundationPose", vis_bgr)

            if writer is not None:
                writer.write(vis_bgr)

            if frame_idx % 50 == 0:
                print(f"[信息] 已处理 {frame_idx} 帧 …")
        else:
            if show_gui:
                cv2.imshow("FoundationPose", cv2.cvtColor(cur_rgb, cv2.COLOR_RGB2BGR))

        if show_gui:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[信息] 用户退出。")
                break
            elif key == ord("r"):
                print("[信息] 重新初始化 …")
                tracker.reset()
                initialized = False

        frame_idx += 1
        cur_rgb, cur_depth = source.read()

    # ── 清理 ──
    source.release()
    if writer is not None:
        writer.release()
    if show_gui:
        cv2.destroyAllWindows()
    summary = f"[信息] 共处理 {frame_idx} 帧"
    if args.save_pose:
        summary += f"，姿态矩阵已保存至 {pose_dir}/"
    if args.save_vis:
        summary += f"，可视化图片已保存至 {vis_dir}/"
    print(summary)


if __name__ == "__main__":
    main()
