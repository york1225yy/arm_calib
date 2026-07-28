# FoundationPose 位姿估计 API 使用说明

本文档说明如何在**其他脚本 / 模块**中直接 `import` [foundationpose_api.py](foundationpose_api.py)，
调用其中封装好的 `PoseEstimatorAPI` 完成 6-DoF 目标位姿估计，涵盖**单帧调用**与
**摄像头/视频连续帧调用**两种典型场景，以及同一帧内多个目标（`MultiObjectPoseEstimator`，
详见第 8 节）的估计方式，以及所有相关输入/输出参数的说明。

> 如果只是想跑现成的 CLI 脚本（视频/摄像头实时可视化），请直接使用
> `python foundationpose_api.py --mesh ... --input ...`（见文件头部 docstring），
> 本文档只针对**代码里直接调用 API** 的场景。

---

## 1. 核心概念：两个类的关系

`foundationpose_api.py` 里有两层接口，其他代码通常只需要用 **`PoseEstimatorAPI`**：

```mermaid
flowchart TD
    A[PoseEstimatorAPI<br/>推荐：面向调用方的简洁接口] --> B[FoundationPoseTracker<br/>内部封装：管理权重/网格/跟踪状态]
    B --> C["FoundationPose.register()<br/>（首帧/独立估计，需要 mask）"]
    B --> D["FoundationPose.track_one()<br/>（后续帧增量跟踪，不需要 mask）"]
```

| 方法 | 对应底层调用 | 是否需要 mask | 是否依赖上一帧 | 速度 | 用途 |
|---|---|---|---|---|---|
| `estimate()` | `register()` | 需要 | 不依赖（可选清空历史状态） | 较慢（含旋转假设采样 + 打分选优） | 首帧初始化 / 单帧独立估计 / 跟踪丢失后重新初始化 |
| `track()` | `track_one()` | 不需要 | 依赖（用上一帧姿态作为起点精化） | 更快 | 摄像头/视频连续帧的实时跟踪 |

> 若同一帧图像里需要同时估计多个不同目标（如两个方形螺母 + 一个圆形螺母），
> 请直接跳到第 8 节使用 `MultiObjectPoseEstimator`；本节以及第 2~7 节描述的 `PoseEstimatorAPI`
> 只针对单个目标。

---

## 2. 准备工作

```python
import numpy as np
from foundationpose_api import PoseEstimatorAPI

# 相机内参 3x3 矩阵（可从标定结果 / cam_K.txt 加载）
K = np.loadtxt("pose_estimation_data/cam_K.txt").reshape(3, 3)

# 构造一次 estimator：会加载 CAD 网格 + 权重（ScorePredictor / PoseRefinePredictor），
# 耗时数秒级，只应做一次，不要在循环里重复构造。
estimator = PoseEstimatorAPI(
    mesh_file="nut_mesh/textured_simple.obj",   # 目标物体 CAD 网格
    K=K,
)
```

依赖环境：需要 CUDA + 编译好的 FoundationPose 扩展（`estimater`/`Utils`/`nvdiffrast`），
且需要设置环境变量 `FOUNDATIONPOSE_ROOT` 指向 FoundationPose 源码根目录（或将本文件
放在其源码目录下），详见 [foundationpose_api.py](foundationpose_api.py) 头部说明。

---

## 3. 单帧调用（独立估计，互不依赖）

适用场景：离线批处理、关键帧逐帧独立估计（如本仓库 [05_collect_pose_estimation_data_mujoco.py](05_collect_pose_estimation_data_mujoco.py) 采集的离散关键帧）、单目标精度验证脚本
（如 [08_compute_nut_pose_in_base.py](08_compute_nut_pose_in_base.py) 单螺母场景下的默认模式；该脚本现已支持多目标，
见第 8 节）等。

```python
# rgb   : uint8, (H, W, 3)，RGB 通道顺序（不是 BGR！）
# mask  : uint8 / bool, (H, W)，>0 或 True 表示目标区域
# depth : float32, (H, W)，单位：米（可选，不给则用假设深度，z 平移不准）
pose_4x4 = estimator.estimate(rgb=rgb_img, mask=mask_img, depth=depth_img)

# pose_4x4: np.ndarray, shape (4, 4), float64
#   物体坐标系 -> 相机坐标系（OpenCV 约定：+X右 +Y下 +Z前）的齐次变换矩阵
```

`estimate()` 默认 `reinit=True`：**每次调用前都会先清空内部跟踪状态**，保证本次
估计与上一次调用完全独立、互不影响。适合下面这种"对多张互不相关的图片分别求位姿"
的用法：

```python
for rgb, mask, depth in dataset:               # 每一份数据互不相关
    pose = estimator.estimate(rgb, mask, depth)  # 每次都是全新独立估计
    ...
```

---

## 4. 摄像头 / 视频连续帧调用（首帧估计 + 后续帧跟踪）

适用场景：对着摄像头或视频实时/连续估计同一个物体的运动轨迹，追求速度。
典型模式是"首帧（或需要重新初始化时）用 `estimate()`，后续帧用 `track()`"：

```python
initialized = False

while True:
    rgb, depth = read_next_frame()   # 你自己的取流逻辑

    if not initialized:
        mask = get_mask_for_first_frame(rgb)   # 首帧需要 mask（比如分割网络/人工框选）
        pose = estimator.estimate(rgb, mask=mask, depth=depth, reinit=False)
        initialized = True
    else:
        pose = estimator.track(rgb, depth=depth)   # 后续帧：不需要 mask，速度更快

    # pose: (4, 4)，物体在当前帧相机坐标系下的位姿
    consume(pose)
```

**关键点：首帧调用 `estimate()` 时要传 `reinit=False`**，这样估计结果会正常写入
内部跟踪状态，紧接着才能调用 `track()` 继续跟踪；如果用默认的 `reinit=True`，
效果和单帧独立估计一样（但因为 `estimate()` 本身内部会先做一次 register 再把结果
写入状态，实际上后面依然可以 `track()`——`reinit` 影响的只是"调用前是否清空历史
状态"，不影响"调用后是否更新状态"）。简单记忆：

- **只做单帧独立估计**：`estimator.estimate(rgb, mask, depth)`（用默认值即可）。
- **视频/摄像头连续帧**：首帧或需要重新初始化时 `estimator.estimate(rgb, mask, depth)`，
  之后每帧 `estimator.track(rgb, depth)`。

### 4.1 跟踪丢失 / 需要重新初始化

`track()` 是增量式的，如果物体被遮挡、跳变或者跟丢了，需要重新提供 mask 走一次
`estimate()` 重新初始化：

```python
if pose_confidence_too_low(pose):
    mask = get_mask_again(rgb)          # 重新分割/框选
    pose = estimator.estimate(rgb, mask=mask, depth=depth)
```

也可以显式调用 `estimator.reset()` 清空状态（不会立即触发估计，只是让下一次
`estimate()` 视为全新的独立估计；`track()` 在 reset 后、estimate 前调用会报错）。

### 4.2 判断是否已经初始化

```python
if estimator.is_initialized:
    pose = estimator.track(rgb, depth)
else:
    pose = estimator.estimate(rgb, mask, depth)
```

---

## 5. 参数说明

### 5.1 `PoseEstimatorAPI.__init__(...)`（只需要构造一次）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `mesh_file` | `str` | 必填 | 目标物体 CAD 网格路径（`.obj`/`.ply`/`.stl` 等 trimesh 支持的格式）。本仓库螺母网格见 [nut_mesh/textured_simple.obj](nut_mesh/textured_simple.obj)，由 [06_convert_nut_mesh_for_foundationpose.py](06_convert_nut_mesh_for_foundationpose.py) 生成，局部坐标系与仿真场景中 `square_nut` body 坐标系严格对齐。 |
| `K` | `np.ndarray (3,3)` | 必填 | 相机内参矩阵。 |
| `weights_dir` | `str` | 本文件同级 `weights/` | 预训练权重目录，内含 `refiner` / `scorer` 子目录。 |
| `est_refine_iter` | `int` | `5` | `estimate()`（register）的精化迭代次数，越大越准但越慢。 |
| `track_refine_iter` | `int` | `2` | `track()`（track_one）的精化迭代次数，跟踪阶段帧间运动通常较小，可以用更少迭代。 |
| `debug` | `int` | `0` | 调试级别：0=关闭，1=显示，2=保存中间调试文件。 |
| `debug_dir` | `str` | `"output"` | 调试文件保存目录（`debug>=1` 时生效）。 |

### 5.2 `estimator.estimate(rgb, mask, depth=None, reinit=True)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `rgb` | `uint8, (H, W, 3)` | **RGB** 通道顺序（不是 OpenCV 默认的 BGR，如果用 `cv2.imread` 读图，需要先 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`）。 |
| `mask` | `uint8 / bool, (H, W)` | 目标物体二值掩码，`>0` 或 `True` 表示目标区域，其余为背景。掩码质量直接影响估计精度。 |
| `depth` | `float32, (H, W)`，可选 | 单位**米**。不提供时用固定假设深度填充整张图（旋转估计仍有效，但 z 方向平移不可信，仅用于粗略调试）。 |
| `reinit` | `bool`，默认 `True` | `True`：调用前清空历史跟踪状态，本次视为完全独立的估计；`False`：不清空，估计结果正常更新跟踪状态，可紧接着调用 `track()`。 |

返回：`np.ndarray, float64, (4, 4)` —— 物体坐标系 → 相机坐标系（OpenCV 约定：+X右
+Y下 +Z前）的齐次变换矩阵，即 $p_{cam} = T \cdot p_{obj}$。

### 5.3 `estimator.track(rgb, depth=None)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `rgb` | `uint8, (H, W, 3)` | 同上，RGB 顺序。 |
| `depth` | `float32, (H, W)`，可选 | 同上，单位米。 |

调用前必须已经通过 `estimate()` 完成过至少一次初始化，否则抛出 `RuntimeError`。
返回值格式同 `estimate()`。

### 5.4 其他方法

| 方法 | 说明 |
|---|---|
| `estimator.is_initialized` | 属性，`bool`，是否已经完成过至少一次 `estimate()`（可以开始调用 `track()`）。 |
| `estimator.reset()` | 清空内部跟踪状态，下一次 `estimate()` 视为全新的独立估计。 |
| `estimator.visualize(rgb, pose)` | 在图像上绘制 3D 包围框与坐标轴，返回 `uint8 BGR` 图像，便于 `cv2.imshow`/`cv2.imwrite` 调试可视化。 |

### 5.5 一次性便捷函数：`estimate_6dof_pose(...)`

如果只想调用一次、不想手动管理 `PoseEstimatorAPI` 实例，可以用这个免维护的自由函数
（内部会临时创建一个 `PoseEstimatorAPI`，用完即弃）：

```python
from foundationpose_api import estimate_6dof_pose

pose_4x4 = estimate_6dof_pose(
    mesh_file="nut_mesh/textured_simple.obj",
    rgb=rgb_img, mask=mask_img, K=K, depth=depth_img,
)
```

**注意**：权重和网格加载本身较慢（秒级），如果需要连续/多次调用，请直接使用
`PoseEstimatorAPI` 类并复用同一个实例，避免每次调用都重新加载。

---

## 6. 输出结果如何使用

`estimate()` / `track()` 返回的都是**物体在相机坐标系下的位姿** `T_obj_cam`
（4×4 齐次变换矩阵）。如果需要进一步换算到机械臂基坐标系（例如做抓取），
需要结合手眼关系和机械臂正向运动学做矩阵连乘，完整示例见
[08_compute_nut_pose_in_base.py](08_compute_nut_pose_in_base.py)：

```python
T_nut_base = T_gripper_base @ T_cam_gripper @ T_nut_cam
```

其中 `T_nut_cam` 就是本 API 的输出结果，`T_cam_gripper`（相机相对末端法兰的固定
安装关系）和 `T_gripper_base`（由机械臂关节角正向运动学计算）来自机器人本体，
不属于 FoundationPose API 的职责范围。

若同一帧中需要对**多个目标**分别做这一整套换算（如 [gen3_with_two_nuts_and_round_nut.xml](gen3_with_two_nuts_and_round_nut.xml)
场景中的 3 个螺母），`T_cam_gripper`/`T_gripper_base` 同一帧内所有目标共用（只取决于机械臂
关节角，与目标无关），只有 `T_nut_cam` 需要对每个目标分别估计，可直接用
`MultiObjectPoseEstimator`（第 8 节）代替 `PoseEstimatorAPI` 对每个目标分别调用，具体完整
示例见 [08_compute_nut_pose_in_base.py](08_compute_nut_pose_in_base.py) 的 `run_multi_object()`。

---

## 7. 常见问题

- **首帧一定要给 mask 吗？** 是的，`estimate()`（register）必须依赖 mask 确定目标
  的大致位置/朝向初值；`track()` 不需要 mask，因为它复用上一帧的姿态作为起点。
- **`reinit` 到底该传 True 还是 False？** 如果调用之后还想紧接着 `track()`
  连续跟踪，传 `False`；如果只是想做一次孤立的估计（比如批量处理很多不相关的图片），
  用默认的 `True` 即可。
- **`depth` 可以不传吗？** 可以，但 z 方向平移会不准确（仅用假设的固定深度），
  只建议在没有深度相机、只关心朝向的场景下使用。
- **多次调用会不会很慢？** 构造 `PoseEstimatorAPI(...)` 本身（加载权重、网格）
  较慢，应该只做一次；之后反复调用 `estimate()`/`track()` 不会重新加载，速度
  取决于 `est_refine_iter`/`track_refine_iter` 迭代次数。
- **场景里有好几个目标（比如 2 个方形螺母 + 1 个圆形螺母），怎么办？** 见下面
  第 8 节的 `MultiObjectPoseEstimator`，以及 [08_compute_nut_pose_in_base.py](08_compute_nut_pose_in_base.py)
  的多目标模式（`--object_mesh`/`--object_grasp_pose_file` 按目标名单独指定网格/抓取位姿）。

---

## 8. 多目标位姿估计（同一帧内多个不同目标）

适用场景：同一帧图像里有多个待估计的物体实例（可以是同一 mesh 的多个不同姿态实例，
如 [gen3_with_two_nuts.xml](gen3_with_two_nuts.xml) 里的两个方形螺母；也可以是不同 mesh
的不同物体，如 [gen3_with_two_nuts_and_round_nut.xml](gen3_with_two_nuts_and_round_nut.xml)
里 2 个方形螺母 + 1 个圆形螺母）。每个目标各自维护独立的姿态/跟踪状态，但共享同一份
权重（`ScorePredictor`/`PoseRefinePredictor`）和 CUDA 光栅化上下文，避免多个目标重复
加载权重带来的显存/时间开销。

### 8.1 基本用法

```python
import cv2
import numpy as np
from foundationpose_api import MultiObjectPoseEstimator

K = np.loadtxt("pose_estimation_data_3nuts/cam_K.txt").reshape(3, 3)

# {目标名: mesh 文件路径}：目标名对应采集数据 masks/<目标名>/ 子目录名，
# 多个目标名可以指向同一个 mesh 文件（如两个方形螺母共用同一个 mesh）。
estimator = MultiObjectPoseEstimator(
    objects={
        "square_nut":   "nut_mesh/textured_simple.obj",
        "square_nut_2": "nut_mesh/textured_simple.obj",
        "round_nut":    "nut_mesh/round_nut_textured_simple.obj",
    },
    K=K,
)

rgb = ...      # (H, W, 3) uint8, RGB 顺序，同一帧内所有目标共用
depth = ...    # (H, W) float32, 单位米，可选，同一帧内所有目标共用
masks = {
    "square_nut":   cv2.imread("pose_estimation_data_3nuts/masks/square_nut/frame_000000.png", cv2.IMREAD_GRAYSCALE),
    "square_nut_2": cv2.imread("pose_estimation_data_3nuts/masks/square_nut_2/frame_000000.png", cv2.IMREAD_GRAYSCALE),
    "round_nut":    cv2.imread("pose_estimation_data_3nuts/masks/round_nut/frame_000000.png", cv2.IMREAD_GRAYSCALE),
}

# 首帧 / 需要重新初始化：对提供了有效掩码的目标分别执行一次完整估计
poses = estimator.estimate_all(rgb, masks, depth=depth)
# poses: dict[str, np.ndarray(4,4)]，只包含本次成功估计的目标

# 后续帧：不需要 mask，对已初始化的目标分别做增量跟踪
poses = estimator.track_all(rgb, depth=depth)

# 可视化：把所有目标的 3D 包围框 + 坐标轴 + 名称标签叠加到同一张图上
vis_bgr = estimator.visualize_all(rgb, poses)
cv2.imwrite("output_3nuts/vis/000000.png", vis_bgr)
```

### 8.2 `MultiObjectPoseEstimator` 参数说明

`__init__(objects, K, weights_dir=..., est_refine_iter=5, track_refine_iter=2, debug=0, debug_dir="output")`

| 参数 | 类型 | 说明 |
|---|---|---|
| `objects` | `dict[str, str]` | `{目标名: mesh 文件路径}`。目标名是调用方自定义的唯一标识（通常对应采集数据 `masks/<目标名>/` 子目录名），多个目标名可指向同一个 mesh 文件。 |
| `K`/`weights_dir`/`est_refine_iter`/`track_refine_iter`/`debug`/`debug_dir` | 同 `PoseEstimatorAPI` | 作用于内部所有目标共用。 |

| 方法 | 说明 |
|---|---|
| `estimate_all(rgb, masks, depth=None, reinit=True)` | 对 `masks` 中提供了有效掩码（非 `None` 且非全 0）的目标分别执行一次 `estimate()`；名称不在 `objects` 中的条目会被忽略并打印警告。返回 `dict[名称, pose_4x4]`，只包含本次成功估计的目标。 |
| `track_all(rgb, depth=None, only_initialized=True, exclude=None)` | 对已初始化过的目标分别执行 `track()`；`exclude` 可传入本次要跳过的目标名集合（如同一帧里刚用 `estimate_all()` 重新初始化过的目标）。返回 `dict[名称, pose_4x4]`。 |
| `visualize_all(rgb, poses, colors=None)` | 把 `poses` 中每个目标的 3D 包围框/坐标轴叠加绘制到同一张图上，并在每个目标投影中心附近标注名称。返回 `uint8 BGR` 图像。 |
| `is_initialized(name)` | 指定目标是否已完成过至少一次 `estimate_all()`。 |
| `reset(name=None)` | 重置指定目标（或 `name=None` 时重置全部目标）的跟踪状态。 |
| `object_names` | 属性，所有已注册的目标名称列表。 |

### 8.3 命令行多目标模式

`foundationpose_api.py` 的 CLI 会自动检测 `--mask_dir`（或 `<input>/masks/`）下是否为
「每个目标一个子目录」的结构，是则自动进入多目标模式：

```bash
python foundationpose_api.py \
    --mesh nut_mesh/textured_simple.obj \
    --object_mesh round_nut=nut_mesh/round_nut_textured_simple.obj \
    --input pose_estimation_data_3nuts --cam_K_file pose_estimation_data_3nuts/cam_K.txt \
    --output_dir output_3nuts --save_vis
```

`--mesh` 是未被 `--object_mesh` 覆盖的目标的默认网格（这里两个方形螺母共用）；
`--object_mesh NAME=PATH` 可重复传入，为指定目标单独指定网格（这里 `round_nut`
使用圆形螺母网格）。各目标的姿态矩阵分别保存至 `output_3nuts/poses/<目标名>/*.txt`，
叠加可视化结果保存至 `output_3nuts/vis/*.png`。

### 8.4 结合机械臂运动学换算到基坐标系（多目标版）

[08_compute_nut_pose_in_base.py](08_compute_nut_pose_in_base.py) 会自动检测
`<data_dir>/masks/` 是否为多目标结构，是则调用其内部的 `run_multi_object()`，
对每个目标分别算出完整的变换链（`T_nut_cam` → `T_nut_base` → `T_grasp_base` →
`T_flange_base`）并与该目标的仿真真值比较，最后汇总跨目标的整体精度统计：

```bash
python 08_compute_nut_pose_in_base.py \
    --xml gen3_with_two_nuts_and_round_nut.xml \
    --mesh nut_mesh/textured_simple.obj \
    --object_mesh round_nut=nut_mesh/round_nut_textured_simple.obj \
    --grasp_pose_file nut_grasp_pose.json \
    --object_grasp_pose_file round_nut=nut_grasp_pose_round.json \
    --data_dir pose_estimation_data_3nuts --frame_idx 0 \
    --tcp_flange_file tcp_flange.json \
    --save_result output_3nuts/nut_pose_in_base_000000.json
```

其中 `--grasp_pose_file`/`--object_grasp_pose_file` 的关系与 `--mesh`/`--object_mesh`
完全一致：未被 `--object_grasp_pose_file` 覆盖的目标使用 `--grasp_pose_file` 指定的
公共抓取位姿文件。两个方形螺母的把手几何完全相同，可以共用
[nut_grasp_pose.json](nut_grasp_pose.json)；圆形螺母的把手中心位置与方形螺母不同
（局部 X 方向 0.06 而非 0.054），需要单独提供
[nut_grasp_pose_round.json](nut_grasp_pose_round.json)。

