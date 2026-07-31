#!/usr/bin/env python3
"""
grasp_planner.py
-----------------
【抓取规划控制算法模块】——从 10_pick_and_place_nuts_mujoco.py 中拆分出来
的"抓取规划/控制"部分：给定感知模块算出的目标位置（如
nut_perception.NutPerceptionModule.compute_T_flange_base() 的结果），
求解到达该位置所需的机械臂关节角（IK），并规划/执行"移动到目标 -> 返回
初始姿态"的运动队列。

设计目的
--------
本文件完全不依赖 FoundationPose / 感知相关代码（不 import
foundationpose_api、不 import nut_perception），只依赖 sim_common.py
里的纯几何/MuJoCo 控制原语；感知模块产出的"目标位置"以普通 numpy 数组
的形式传入即可。这样只做抓取规划/控制算法开发的同事，只需要修改本文件，
完全不需要关心 FoundationPose 推理细节或 UI/渲染代码。

包含内容：
  - solve_ik_position()：只约束末端 xyz 位置、不约束姿态的阻尼最小二乘
    （DLS）雅可比逆运动学求解器；
  - MotionQueuePlanner：管理"依次移动到多个目标位置、每次到达后返回初始
    姿态"的运动队列（关节空间线性插值 + 到位停留 + 自动切换下一目标）。
"""

import numpy as np
import mujoco

from sim_common import ARM_JOINT_NAMES, FLANGE_BODY_NAME, get_arm_qpos, set_arm_qpos, set_arm_ctrl


def solve_ik_position(model, ik_data, target_pos, q0, body_name=FLANGE_BODY_NAME,
                       max_iter=300, pos_tol=1e-3, damping=1e-3, step_clip=0.2):
    """阻尼最小二乘（DLS）雅可比逆运动学，只约束位置（3维），不约束姿态。

    求解 7 个关节角，使 body_name（默认末端法兰 "bracelet_link"）在世界/
    机械臂坐标系下的位置逼近 target_pos。之所以只约束 xyz 位置、完全不管
    姿态：法兰的目标姿态若同时与感知模块算出的旋转部分严格匹配，在某些
    螺母朝向下可能无解或收敛困难（该姿态是由"抓取偏移 T_grasp_nut"直接
    复合出来的，不保证一定在机械臂的可达姿态范围内）；只求位置解，忽略
    姿态误差，求解更容易收敛、更稳定，足以验证/演示"感知算出的目标位置
    是否正确"。

    q0 为迭代初值（热启动）。ik_data 为独立于真实仿真 data 的 scratch
    MjData，反复调用 mj_forward/mj_jacBody 做正向运动学试算，不会扰动
    真正在运行物理仿真的 data。
    """
    set_arm_qpos(model, ik_data, q0)
    mujoco.mj_forward(model, ik_data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"找不到 body: {body_name}")
    arm_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES]
    dof_idx = [model.jnt_dofadr[j] for j in arm_jids]

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(max_iter):
        pos = ik_data.xpos[body_id].copy()
        pos_err = target_pos - pos

        if np.linalg.norm(pos_err) < pos_tol:
            break

        mujoco.mj_jacBody(model, ik_data, jacp, jacr, body_id)
        J = jacp[:, dof_idx]  # (3, 7)，只取平移部分的雅可比

        JJt = J @ J.T + damping * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, pos_err)
        dq = np.clip(dq, -step_clip, step_clip)

        q = get_arm_qpos(model, ik_data)
        set_arm_qpos(model, ik_data, q + dq)
        mujoco.mj_forward(model, ik_data)

    return get_arm_qpos(model, ik_data)


class MotionQueuePlanner:
    """管理"依次移动到多个目标位置、每次到达后返回初始姿态"的运动队列。

    使用方式：
        planner = MotionQueuePlanner(model, ik_data, q_start,
                                      move_seconds=2.5, hold_seconds=1.0, fps=30)
        planner.build_queue({"square_nut": target_xyz1, "round_nut": target_xyz2},
                             q_seed=current_q)
        ...
        # 每个显示帧调用一次：
        new_q = planner.update(current_q_manual)
        if new_q is not None:
            set_arm_ctrl(model, data, new_q)

    关节空间线性插值实现平滑运动（真实物理仿真持续 mj_step，因此手臂会
    平滑地运动过去，而不是瞬间跳变），到位后停留 hold_seconds，再自动
    切换到队列中的下一个目标（先"去程"再"回程"，一去一回为一组，按传入
    targets 的顺序逐组执行）。
    """

    def __init__(self, model, ik_data, q_start, move_seconds=2.5, hold_seconds=1.0, fps=30):
        self.model = model
        self.ik_data = ik_data
        self.q_start = np.array(q_start, dtype=float).copy()
        self.total_steps = max(1, round(move_seconds * fps))
        self.hold_frames = max(0, round(hold_seconds * fps))

        self.queue = []          # 待执行的 (标签, 目标关节角 q7) 列表，先进先出
        self.start_q = None      # 当前运动段的起始关节角
        self.target_q = None     # 当前运动段的目标关节角（None=空闲）
        self.step = 0
        self.hold_counter = 0
        self.label = None

    @property
    def is_idle(self):
        return self.target_q is None and not self.queue

    @property
    def queue_remaining(self):
        return len(self.queue)

    def build_queue(self, targets, q_seed, body_name=FLANGE_BODY_NAME):
        """targets: dict[名称, 目标位置 xyz(3,)]，按 dict 迭代顺序依次求解
        IK 并规划"去程 -> 回程(回到 q_start)"。q_seed 为第一个目标 IK 求解
        的初值（通常传入当前 q_manual）；之后每个目标都从 q_start 热启动，
        保证多个目标之间的 IK 解相互独立、不受求解顺序影响。

        返回 dict[名称, IK解 q7]，供调用方打印/记录。"""
        if not self.is_idle:
            raise RuntimeError("上一轮运动尚未执行完毕，请等待完成后再规划新队列。")

        q_seed = np.array(q_seed, dtype=float).copy()
        queue = []
        ik_solutions = {}
        for name, target_pos in targets.items():
            q_ik = solve_ik_position(self.model, self.ik_data, target_pos, q_seed, body_name=body_name)
            ik_solutions[name] = q_ik
            queue.append((f"go->{name}", q_ik))
            queue.append((f"return<-{name}", self.q_start.copy()))
            q_seed = self.q_start.copy()  # 每次都从初始姿态热启动下一次 IK，保证解的一致性

        self.queue = queue
        self._advance(q_current=self.q_start)
        return ik_solutions

    def _advance(self, q_current):
        if not self.queue:
            self.target_q = None
            self.label = None
            return
        self.label, self.target_q = self.queue.pop(0)
        self.start_q = np.array(q_current, dtype=float).copy()
        self.step = 0
        self.hold_counter = 0
        print(f"    [运动规划] 开始执行: {self.label}")

    def update(self, q_current):
        """每个显示帧调用一次：若运动队列非空，返回本帧应下发的新关节角
        （调用方负责 set_arm_ctrl）；若队列为空/已完成，返回 None。"""
        if self.target_q is None:
            return None
        t = min(1.0, (self.step + 1) / self.total_steps)
        q_new = self.start_q + (self.target_q - self.start_q) * t
        self.step += 1
        if t >= 1.0:
            self.hold_counter += 1
            if self.hold_counter >= self.hold_frames:
                self._advance(q_current=q_new)
        return q_new
