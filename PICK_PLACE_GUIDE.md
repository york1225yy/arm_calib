# MuJoCo 螺母抓取 Demo 使用与开发指南

本指南面向 `arm_calib/10_pick_and_place_nuts_mujoco.py` 及其依赖的
`arm_calib/pick_place_modules/` 模块，包含两部分内容：

1. **完整功能与使用方法**：这套代码是做什么的、怎么运行、按键/参数含义。
2. **二次开发指南**：如果要修改/增加"感知"或"抓取规划控制"部分的功能，
   应该改哪个文件、哪个函数，以及各部分之间的边界在哪里。

---

## 1. 系统概览

这是一个基于 **MuJoCo 仿真 + FoundationPose 6D 位姿估计** 的 Kinova Gen3
机械臂抓取螺母 Demo：仿真场景里桌面上放着若干螺母，头顶固定一个 RGB-D
相机（D435i），程序读取相机图像、用 FoundationPose 估计螺母的 6D 位姿，
再结合手眼标定结果和抓取姿态标定文件，把螺母位姿换算到机械臂基座坐标系
下的目标法兰位姿，最后用逆运动学 (IK) 求解关节角并驱动机械臂运动过去
"抓取"（当前 demo 只做运动到位，不含真实力控夹爪）。

### 代码结构（职责边界）

```
arm_calib/
├── 10_pick_and_place_nuts_mujoco.py   ← UI/编排层（薄）
└── pick_place_modules/
    ├── sim_common.py       ← 公共基础设施（双方都可能用到）
    ├── nut_perception.py   ← 感知算法模块（只负责"螺母在哪"）
    └── grasp_planner.py    ← 抓取规划/控制算法模块（只负责"怎么移动过去"）
```

| 文件 | 职责 | 谁应该改它 |
|---|---|---|
| `10_pick_and_place_nuts_mujoco.py` | 命令行参数、加载模型、渲染显示、键盘交互状态机、录像、主循环 | 只有新增/调整按键、显示内容、整体流程时才改；**不应在此写感知或规划算法本身** |
| `pick_place_modules/sim_common.py` | 常量（关节名、相机名、包围盒尺寸…）、坐标变换工具、MuJoCo 控制原语、标定文件加载、OpenCV 3D 可视化 | 感知/规划双方共用的基础能力都放这里 |
| `pick_place_modules/nut_perception.py` | `NutPerceptionModule` 类：调用 FoundationPose、位姿合理性过滤、坐标换算、真值对比 | **负责感知算法的人改这里** |
| `pick_place_modules/grasp_planner.py` | `solve_ik_position()` + `MotionQueuePlanner`：IK 求解、运动队列/插值 | **负责抓取规划控制算法的人改这里** |
| `foundationpose_api.py` | FoundationPose 底层封装（`MultiObjectPoseEstimator` 等） | 只有需要改 FoundationPose 底层调用方式时才改，`nut_perception.py` 不应重复实现这里已有的逻辑 |

两个模块 (`nut_perception.py` / `grasp_planner.py`) 彼此**没有依赖关系**
（`grasp_planner.py` 完全不 import 任何感知相关代码），因此可以完全并行开发。

---

## 2. 完整使用方法

### 2.1 运行方式

```bash
# 交互模式：弹出 OpenCV 窗口，手动按键操作
python 10_pick_and_place_nuts_mujoco.py --viewer

# 无头自动化模式（用于测试/CI）：不弹窗，按固定帧数自动触发估计+打印
python 10_pick_and_place_nuts_mujoco.py --no_gui --auto

# 交互 + 保存录像
python 10_pick_and_place_nuts_mujoco.py --viewer --save_video output/demo.mp4
```

### 2.2 交互流程（状态机）

程序有三种显示/运行状态：`idle` → `tracking` 或 `ground_truth`。

1. 启动后先 `settle()`：运行 `--settle_steps`（默认 800）步物理仿真，
   让螺母在重力下自然落稳，再进入主循环，此时处于 `idle` 状态（只显示
   原始相机画面）。
2. 按 **`e`**：从 `idle` 进入 `tracking` 状态——每帧调用 FoundationPose
   （异步估计 + 增量跟踪），画面上叠加显示：
   - 黄色 3D 框：算法估计出的螺母位姿（`pose source: algorithm estimation`）
   - 绿色 3D 框：换算出的法兰目标位姿
3. 按 **`t`**：从 `idle` 进入 `ground_truth` 状态——**不跑算法**，直接读取
   仿真里螺母 body 的真实位姿作为"位姿"，用于验证坐标换算/IK/运动链路
   本身是否正确（排除感知误差）。画面上用青色框标注
   （`pose source: simulation ground truth`）。
4. 按 **`r`**：在 `tracking`/`ground_truth` 状态下重新初始化（清空当前
   估计状态，重新来一次）。
5. 按 **`p`**：打印所有 `--nut_order` 中螺母的 `T_grasp_base` /
   `T_flange_base` 位姿，以及（仅仿真里可用）与真值的位置/姿态误差。
   要求所有螺母都已经有有效位姿。
6. 按 **`g`**：根据当前螺母法兰目标位姿构建一条运动队列（对每个螺母依次
   IK 求解 → 走过去 → 停留 → 返回起始姿态 → 走下一个），此后主循环每帧
   自动推进这条队列直到走完。
7. 按 **`1`-`7`**：选中对应关节做手动控制，配合 **`[`** / **`]`** 减小/
   增大该关节角度；**`c`** / **`o`** 手动开合夹爪（当前 XML 里夹爪执行器
   被注释掉，实际不生效，接口保留）。
8. 按 **`q`**：退出。

`--auto` 模式会在 `--auto_estimate_after`（默认第 30 帧）自动模拟按
`e`，在 `--auto_print_after`（默认第 60 帧）自动模拟按 `p`，用于无 GUI
环境下的冒烟测试。

### 2.3 主要命令行参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--xml` | `gen3_with_nut.xml` | MuJoCo 场景文件 |
| `--start_pose_file` | — | 机械臂起始关节角文件 |
| `--mesh` / `--round_mesh` | — | 方形/圆形螺母 mesh 路径（供 FoundationPose 用） |
| `--grasp_pose_file` / `--round_grasp_pose_file` | — | 抓取姿态标定文件（螺母坐标系下的抓取位姿候选） |
| `--tcp_flange_file` | — | TCP→法兰的标定变换文件 |
| `--weights_dir` | `weights/` | FoundationPose 权重目录 |
| `--nut_order` | `square_nut,square_nut_2,round_nut` | 参与估计/抓取的螺母名称及顺序（对应 `sim_common.NUT_BODY_NAMES`） |
| `--settle_steps` | 800 | 启动后物理沉降步数 |
| `--est_refine_iter` / `--track_refine_iter` | 5 / 2 | FoundationPose 初始估计/跟踪的迭代精修次数 |
| `--move_seconds` / `--hold_seconds` | 2.5 / 1.0 | 运动队列每段的移动/停留时长 |
| `--fps` | 30 | 显示与录像帧率 |
| `--save_video` | — | 保存 mp4 路径 |
| `--viewer` | False | 是否弹窗显示 |
| `--no_gui` | False | 无头模式 |
| `--auto` / `--auto_estimate_after` / `--auto_print_after` | — | 自动化冒烟测试相关 |

---

## 3. 二次开发指南：如何扩展

### 3.1 扩展 / 修改「感知」算法 → 只改 `pick_place_modules/nut_perception.py`

核心是 `NutPerceptionModule` 类，典型改动点：

- **提高/降低估计触发条件**：模块常量 `MIN_MASK_PIXELS`（掩码最小像素数
  阈值）、`DEPTH_SANITY_RANGE_M`（深度合理范围）。
- **调整位姿合理性过滤逻辑**：函数 `pose_is_plausible()`。
- **修正对称/扁平物体的姿态歧义**（如上下翻转 180°问题）：函数
  `fix_flat_object_updown_ambiguity()`，内含详细调试注释，说明该问题的
  根因和判定方法（局部 Z 轴是否朝下）。
- **修改坐标换算链路**（相机系 → 基座系 → 抓取系 → 法兰系）：
  `compute_T_grasp_base()` / `compute_T_flange_base()`。
- **新增位姿来源**（除了算法估计 `"est"` 和仿真真值 `"gt"` 之外）：
  参考 `get_ground_truth_T_nut_cam()` / `use_ground_truth()` 的写法，
  在 `apply_poses(poses, source=...)` 里统一记录到 `self.pose_source`。
- **调试/精度评估**：`compute_pose_error_vs_ground_truth()`（仅仿真里有
  意义，真实机器人没有真值）。

**不应该做的事**：不要在这个文件里重新实现 FoundationPose 的底层调用
（模型加载、光栅化、ICP 精修等），这些都已经封装在 `foundationpose_api.py`
的 `MultiObjectPoseEstimator` / `FoundationPoseTracker` 里，`nut_perception.py`
只应该调用它的 `estimate_all()` / `track_all()` / `visualize_all()` 接口。
如果确实需要改 FoundationPose 本身的行为（比如迭代次数、光栅化上下文
复用方式），去改 `foundationpose_api.py`。

### 3.2 扩展 / 修改「抓取规划控制」算法 → 只改 `pick_place_modules/grasp_planner.py`

核心是 `solve_ik_position()` 函数和 `MotionQueuePlanner` 类，典型改动点：

- **升级 IK 求解器**：当前 `solve_ik_position()` 是仅约束位置（3-DOF）的
  阻尼最小二乘 (DLS) 雅可比迭代法，故意不约束姿态（详见函数 docstring 里
  的理由：感知给出的抓取姿态不一定运动学可达，位置解更鲁棒、更容易收
  敛）。如果需要完整 6D IK（位置+姿态），在这里新增函数（例如
  `solve_ik_pose()`），保持接口风格一致（输入 `model, ik_data, target,
  q0`，输出 `q_solution`），供 `MotionQueuePlanner` 调用。
- **调整轨迹规划方式**：当前 `MotionQueuePlanner.update()` 是简单的关节
  空间线性插值 + 停留帧数控制。如果要换成梯形速度曲线、三次样条、多点
  路径规划等，改 `build_queue()`（如何生成路径点）和 `update()`（如何
  按时间推进）。
- **扩展运动队列逻辑**（比如加入抓取/释放动作、多阶段路径点）：
  `build_queue()` 里对每个目标追加 `(label, target_q)` 元组的地方，可以
  按需插入更多阶段。

**不应该做的事**：不要在这个文件里 import 任何 `nut_perception` 或
`foundationpose_api` 的内容——这是为了保证抓取规划控制逻辑可以脱离感知
算法独立开发和单测（例如直接给定固定目标位置测试 IK 和运动队列）。
`grasp_planner.py` 只接受"目标位置/位姿"这种通用输入，不关心这个目标是
算法估计出来的还是手工给定的。

### 3.3 共享基础设施 → `pick_place_modules/sim_common.py`

如果双方都需要用到的东西不存在，加在这里，例如：

- 新增/调整常量：关节名列表 `ARM_JOINT_NAMES`、夹爪执行器名
  `GRIPPER_ACTUATOR_NAMES`、末端法兰 body 名 `FLANGE_BODY_NAME`、相机名
  `TOP_CAMERA_NAME`、螺母 body 名列表 `NUT_BODY_NAMES`、可视化包围盒尺寸
  `FLANGE_TARGET_BBOX` / `NUT_POSE_BBOX` 等。
- 新增坐标变换工具（类似 `make_T`、`get_body_T_world`、
  `get_cam_T_world_cv`）。
- 新增 MuJoCo 控制原语（类似 `set_arm_qpos`、`set_arm_ctrl`、
  `set_gripper_ctrl_ratio`）。
- 新增标定文件加载器（类似 `load_start_qpos`、`load_tcp_flange`、
  `load_grasp_pose`）。
- 新增可视化辅助函数（类似 `draw_posed_3d_box_simple`、
  `draw_xyz_axis_simple`）。

### 3.4 主脚本（UI 层）→ `10_pick_and_place_nuts_mujoco.py`

如果要新增按键或显示状态（比如新增一个 `'m'` 键触发某种新模式），改
`Demo.handle_key()`（新增按键分支）和 `Demo.render_and_display()`
（新增显示状态分支），但具体逻辑应该只是**调用**
`self.perception.xxx()` 或 `self.planner.xxx()` 的已有/新增接口方法，
不要在这一层直接写感知算法或 IK/轨迹规划的实现细节。

如果新增了命令行参数，在 `main()` 里的 `argparse` 部分添加，并在
`Demo.__init__()` 里传给对应的 `NutPerceptionModule` 或
`MotionQueuePlanner` 构造函数。

---

## 4. 快速定位速查表

| 我想做的事 | 去改哪里 |
|---|---|
| 螺母位姿估计不准 / 换个检测阈值 | `nut_perception.py` |
| 螺母姿态上下颠倒（180°歧义） | `nut_perception.py` 的 `fix_flat_object_updown_ambiguity()` |
| 相机→基座→抓取→法兰坐标换算有问题 | `nut_perception.py` 的 `compute_T_grasp_base()` / `compute_T_flange_base()` |
| FoundationPose 本身的调用方式/参数 | `foundationpose_api.py` |
| 机械臂走不到目标位置 / IK 不收敛 | `grasp_planner.py` 的 `solve_ik_position()` |
| 想要更平滑/更复杂的运动轨迹 | `grasp_planner.py` 的 `MotionQueuePlanner` |
| 新增一个常量/坐标工具双方都要用 | `sim_common.py` |
| 新增一个按键或显示模式 | `10_pick_and_place_nuts_mujoco.py` 的 `handle_key()` / `render_and_display()` |
| 新增一个命令行参数 | `10_pick_and_place_nuts_mujoco.py` 的 `main()` + `Demo.__init__()` |
