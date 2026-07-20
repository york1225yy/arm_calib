#!/usr/bin/env python3
"""
06_convert_nut_mesh_for_foundationpose.py
--------------------------------------------
将 square-nut.xml（MuJoCo 描述的方形螺母，由 5 个长方体 geom 组成）
转换为 FoundationPose（foundationpose_api.py）可直接加载的 CAD 网格文件
（.obj/.ply/.stl，trimesh 支持的格式）。

重要：本脚本从 square-nut.xml 中解析出的每个 box 的 pos/size 与
gen3_with_nut.xml 中 body "square_nut" 下的 5 个 geom 完全一致（同一组
数值），因此转换出的网格局部坐标系与仿真场景中 square_nut 的 body 坐标系
严格对齐 —— 即网格原点 (0,0,0) 对应仿真里 square_nut 的 body 原点，
姿态估计结果可以直接与仿真真值（如 05_collect_pose_estimation_data_mujoco.py
保存的 nut_pose_world_4x4）比较，无需额外的坐标系换算。

用法：
  python 06_convert_nut_mesh_for_foundationpose.py
  python 06_convert_nut_mesh_for_foundationpose.py --out_dir nut_mesh --format obj

转换后配合 foundationpose_api.py 使用：
  python foundationpose_api.py --mesh nut_mesh/textured_simple.obj \\
      --input pose_estimation_data --cam_K_file pose_estimation_data/cam_K.txt
"""

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
XML_PATH     = os.path.join(os.path.dirname(__file__), "square-nut.xml")
TEXTURE_PATH = os.path.join(os.path.dirname(__file__), "textures", "brass-ambra.png")


def parse_box_geoms(xml_path):
    """解析 square-nut.xml，取出所有 type="box" geom 的 pos（中心）与
    size（半长），二者均为 MuJoCo 原始单位（米）。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    for geom in root.iter("geom"):
        if geom.get("type") != "box":
            continue
        pos  = np.array([float(v) for v in geom.get("pos", "0 0 0").split()])
        size = np.array([float(v) for v in geom.get("size").split()])
        boxes.append((pos, size))

    if not boxes:
        raise ValueError(f"在 {xml_path} 中未解析到任何 box geom")
    return boxes


def get_average_color(texture_path):
    """读取贴图并计算平均颜色（RGB，0~1 浮点），用作网格的漫反射颜色。"""
    import cv2

    if not os.path.isfile(texture_path):
        print(f"[警告] 找不到贴图 {texture_path}，使用默认黄铜色近似值。")
        return np.array([0.65, 0.49, 0.24])  # 黄铜色近似值（RGB）

    bgr = cv2.imread(texture_path, cv2.IMREAD_COLOR)
    mean_bgr = bgr.reshape(-1, 3).mean(axis=0)
    mean_rgb = mean_bgr[::-1] / 255.0
    return mean_rgb


def build_mesh(boxes, rgb_color):
    """将若干个长方体 geom 拼接为单个 trimesh 网格，并赋予统一的漫反射颜色。"""
    import trimesh

    parts = []
    for pos, size in boxes:
        extents = size * 2.0  # MuJoCo box 的 size 是半长，trimesh 用全长
        box = trimesh.creation.box(extents=extents)
        box.apply_translation(pos)
        parts.append(box)

    mesh = trimesh.util.concatenate(parts)

    rgba = np.append(np.clip(rgb_color, 0.0, 1.0) * 255.0, 255.0).astype(np.uint8)
    material = trimesh.visual.material.SimpleMaterial(diffuse=rgba)
    mesh.visual = trimesh.visual.TextureVisuals(material=material)

    return mesh


def main():
    parser = argparse.ArgumentParser(
        description="将 square-nut.xml 转换为 FoundationPose 可用的 CAD 网格文件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xml",     default=XML_PATH, help="square-nut.xml 路径")
    parser.add_argument("--texture", default=TEXTURE_PATH, help="黄铜贴图路径（用于估算网格颜色）")
    parser.add_argument("--out_dir", default="nut_mesh", help="输出目录")
    parser.add_argument("--out_name", default="textured_simple", help="输出文件名（不含扩展名）")
    parser.add_argument("--format",  default="obj", choices=["obj", "ply", "stl"],
                        help="导出网格格式（.obj 会同时生成 .mtl 材质文件）")
    args = parser.parse_args()

    print(f"[解析] {args.xml}")
    boxes = parse_box_geoms(args.xml)
    print(f"  共解析到 {len(boxes)} 个 box geom")

    print(f"[取色] {args.texture}")
    rgb_color = get_average_color(args.texture)
    print(f"  平均颜色 (RGB, 0~1): {rgb_color}")

    print("[构建网格]")
    mesh = build_mesh(boxes, rgb_color)
    print(f"  顶点数: {len(mesh.vertices)}  面片数: {len(mesh.faces)}")
    print(f"  包围盒 (min~max, 米): {mesh.bounds[0]} ~ {mesh.bounds[1]}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.out_name}.{args.format}")
    mesh.export(out_path)
    print(f"\n[完成] 已导出网格: {out_path}")
    print("  姿态估计示例命令：")
    print(f"  python foundationpose_api.py --mesh {out_path} \\")
    print(f"      --input pose_estimation_data --cam_K_file pose_estimation_data/cam_K.txt")


if __name__ == "__main__":
    main()
