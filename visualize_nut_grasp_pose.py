#!/usr/bin/env python3
"""
visualize_nut_grasp_pose.py
----------------------------
使用 MuJoCo 可视化螺母（square_nut）与预设抓取位姿（夹爪 TCP）之间的相对
位姿关系，便于直观检查 nut_grasp_pose.json 中 T_grasp_nut 是否合理
（例如：夹爪是否夹在把手上、闭合方向/接近方向是否会与螺母主体干涉等）。

配套场景文件：nut_grasp_visualization.xml
  - body "nut"：螺母本体（mocap body，默认世界位姿 = 单位阵，即螺母局部坐标系
    与世界坐标系重合）。
  - body "grasp_frame"：简化的二指夹爪示意几何体 + TCP 坐标轴，世界位姿由本
    脚本在启动时计算并写入 data.mocap_pos / data.mocap_quat。

计算关系
--------
    T_grasp_world = T_nut_world @ T_grasp_nut

默认 T_nut_world = 单位阵（螺母放在世界原点，局部坐标系直接当作世界坐标系），
这样最直观地展示"抓取位姿相对螺母局部坐标系"的关系。

如果你想在某个具体世界/基座姿态下检查（例如叠加某一帧的估计结果），可以用
--nut_pose_file 指定一个包含 4x4 矩阵的 json（如 08_compute_nut_pose_in_base.py
的输出 output/nut_pose_in_base_000000.json），并用 --nut_pose_key 选择其中的
键名（默认 "T_nut_base_est"，也可以是 "T_nut_base_gt"）。

用法
----
  # 最简单：只看螺母 + 抓取位姿的相对关系（螺母置于世界原点）
  python visualize_nut_grasp_pose.py

  # 指定抓取位姿文件
  python visualize_nut_grasp_pose.py --grasp_pose_file nut_grasp_pose.json

  # 叠加某一帧估计结果中的螺母世界位姿（可选）
  python visualize_nut_grasp_pose.py \\
      --nut_pose_file output/nut_pose_in_base_000000.json \\
      --nut_pose_key T_nut_base_est

操作说明：
  弹出的 MuJoCo 查看器窗口中可自由拖动旋转/缩放视角（鼠标左键旋转，右键平移，
  滚轮缩放）。窗口关闭即结束程序。场景中：
    - 螺母本体使用与真实模型一致的黄铜材质渲染，红色小球标记螺母中心与把手中心。
    - 黄色小球 = 夹爪 TCP 原点，灰色方块示意二指夹爪（手掌 + 两指）。
    - 红/绿/蓝 圆柱 = 局部坐标系 X/Y/Z 轴（螺母和抓取位姿各有一套）。
"""

import argparse
import json
import os

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 -> MuJoCo 约定的 (w,x,y,z) 四元数。"""
    q_xyzw = Rotation.from_matrix(R).as_quat()  # scipy: (x,y,z,w)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def load_T_grasp_nut(grasp_pose_file: str) -> np.ndarray:
    with open(grasp_pose_file, encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["T_grasp_nut"], dtype=float)


def load_T_nut_world(nut_pose_file, nut_pose_key) -> np.ndarray:
    if nut_pose_file is None:
        return np.eye(4)
    with open(nut_pose_file, encoding="utf-8") as f:
        data = json.load(f)
    if nut_pose_key not in data:
        raise KeyError(
            f"{nut_pose_file} 中未找到键 '{nut_pose_key}'，"
            f"可用键: {list(data.keys())}"
        )
    return np.array(data[nut_pose_key], dtype=float)


def set_mocap_pose(model, data, body_name, T_world: np.ndarray):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"找不到 body: {body_name}")
    mocap_id = model.body_mocapid[body_id]
    if mocap_id < 0:
        raise ValueError(f"body '{body_name}' 不是 mocap body")
    data.mocap_pos[mocap_id] = T_world[:3, 3]
    data.mocap_quat[mocap_id] = rotmat_to_quat_wxyz(T_world[:3, :3])


def print_T(name, T, indent=2):
    pad = " " * indent
    print(f"{name}:")
    for row in T:
        print(pad + "  ".join(f"{v:10.6f}" for v in row))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="可视化螺母与预设抓取位姿（夹爪 TCP）的相对关系",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--xml", default=os.path.join(here, "nut_grasp_visualization.xml"),
        help="场景 xml 文件路径",
    )
    parser.add_argument(
        "--grasp_pose_file", default=os.path.join(here, "nut_grasp_pose.json"),
        help="预设抓取位姿文件（包含 T_grasp_nut）",
    )
    parser.add_argument(
        "--nut_pose_file", default=None,
        help="可选：包含螺母世界位姿的 json（如 08 脚本输出的结果文件）。"
             "省略则螺母置于世界原点（单位阵）。",
    )
    parser.add_argument(
        "--nut_pose_key", default="T_nut_base_est",
        help="--nut_pose_file 中要使用的键名，例如 T_nut_base_est / T_nut_base_gt",
    )
    args = parser.parse_args()

    print(f"[加载场景] {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    T_nut_world = load_T_nut_world(args.nut_pose_file, args.nut_pose_key)
    T_grasp_nut = load_T_grasp_nut(args.grasp_pose_file)
    T_grasp_world = T_nut_world @ T_grasp_nut

    print_T("[T_nut_world]   螺母 -> 世界坐标系", T_nut_world)
    print_T("[T_grasp_nut]   夹爪 TCP -> 螺母局部坐标系（来自预设抓取位姿）", T_grasp_nut)
    print_T("[T_grasp_world] 夹爪 TCP -> 世界坐标系（= T_nut_world @ T_grasp_nut）", T_grasp_world)

    offset_mm = np.linalg.norm(T_grasp_nut[:3, 3]) * 1000.0
    print(f"\n[信息] 夹爪 TCP 与螺母中心距离 ≈ {offset_mm:.1f} mm")

    set_mocap_pose(model, data, "nut", T_nut_world)
    set_mocap_pose(model, data, "grasp_frame", T_grasp_world)
    mujoco.mj_forward(model, data)

    print("\n[信息] 已启动可视化窗口：鼠标左键旋转视角，右键平移，滚轮缩放；关闭窗口结束程序。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 场景为静态（无关节/物理），只需保持窗口打开、定期同步一次即可。
        while viewer.is_running():
            viewer.sync()


if __name__ == "__main__":
    main()
