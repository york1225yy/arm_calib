#!/usr/bin/env python3
"""
06_convert_nut_mesh_for_foundationpose.py
--------------------------------------------
将 square-nut.xml / round-nut.xml（MuJoCo 描述的螺母，均由若干长方体 geom
组成，round-nut.xml 额外用 axisangle 旋转部分 box 拼出八边形近似圆形）
转换为 FoundationPose（foundationpose_api.py）可直接加载的 CAD 网格文件
（.obj/.ply/.stl，trimesh 支持的格式）。

重要：本脚本从 xxx-nut.xml 中解析出的每个 box 的 pos/size/axisangle 与
gen3_with_nut.xml / gen3_with_two_nuts.xml 等场景文件中对应螺母 body 下的
geom 完全一致（同一组数值），因此转换出的网格局部坐标系与仿真场景中该
body 的坐标系严格对齐 —— 即网格原点 (0,0,0) 对应仿真里该螺母 body 的
原点，姿态估计结果可以直接与仿真真值（如
05_collect_pose_estimation_data_mujoco.py 保存的 nut_pose_world_4x4 /
nuts_pose_world_4x4）比较，无需额外的坐标系换算。

用法：
  # 方形螺母（默认）
  python 06_convert_nut_mesh_for_foundationpose.py
  python 06_convert_nut_mesh_for_foundationpose.py --out_dir nut_mesh --format obj

  # 圆形螺母（round-nut.xml，贴图文件 textures/steel-scratched.png 在本仓库中
  # 缺失，脚本会自动退回 --fallback_color 指定的钢灰色近似值）
  python 06_convert_nut_mesh_for_foundationpose.py --xml round-nut.xml \\
      --texture textures/steel-scratched.png --fallback_color "0.75 0.76 0.78" \\
      --out_name round_nut_textured_simple

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
    """解析 nut xml（square-nut.xml / round-nut.xml 等），取出所有
    type="box" geom 的 pos（中心）、size（半长）以及可选的 axisangle
    （绕给定轴的旋转，格式 "ax ay az angle_rad"，round-nut.xml 用它拼出
    八边形近似圆形），均为 MuJoCo 原始单位（米/弧度）。

    返回
    ----
    list[(pos: np.ndarray(3,), size: np.ndarray(3,), axisangle: tuple|None)]
        axisangle 为 None 表示该 box 无额外旋转；否则为 (axis(3,), angle_rad)。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    for geom in root.iter("geom"):
        if geom.get("type") != "box":
            continue
        pos  = np.array([float(v) for v in geom.get("pos", "0 0 0").split()])
        size = np.array([float(v) for v in geom.get("size").split()])
        axisangle_str = geom.get("axisangle")
        axisangle = None
        if axisangle_str:
            vals = [float(v) for v in axisangle_str.split()]
            axisangle = (np.array(vals[:3]), vals[3])  # (axis, angle_rad)
        boxes.append((pos, size, axisangle))

    if not boxes:
        raise ValueError(f"在 {xml_path} 中未解析到任何 box geom")
    return boxes


def get_average_color(texture_path, fallback_rgb):
    """读取贴图并计算平均颜色（RGB，0~1 浮点），用作网格的漫反射颜色。
    找不到贴图文件时使用 fallback_rgb（调用方指定的默认颜色）。"""
    import cv2

    if not os.path.isfile(texture_path):
        print(f"[警告] 找不到贴图 {texture_path}，使用指定的默认颜色近似值 {fallback_rgb}。")
        return np.array(fallback_rgb)

    bgr = cv2.imread(texture_path, cv2.IMREAD_COLOR)
    mean_bgr = bgr.reshape(-1, 3).mean(axis=0)
    mean_rgb = mean_bgr[::-1] / 255.0
    return mean_rgb


def build_mesh(boxes, rgb_color):
    """将若干个长方体 geom 拼接为单个 trimesh 网格，并赋予统一的漫反射颜色。

    每个 box 若带 axisangle，先绕给定轴旋转、再平移到 pos ——
    与 MuJoCo geom 局部变换的语义一致（size/形状在局部系中定义，
    axisangle 描述该 geom 坐标系相对父体的旋转，pos 是平移）。
    """
    import trimesh

    parts = []
    for pos, size, axisangle in boxes:
        extents = size * 2.0  # MuJoCo box 的 size 是半长，trimesh 用全长
        box = trimesh.creation.box(extents=extents)
        if axisangle is not None:
            axis, angle = axisangle
            R = trimesh.transformations.rotation_matrix(angle, axis)
            box.apply_transform(R)
        box.apply_translation(pos)
        parts.append(box)

    mesh = trimesh.util.concatenate(parts)

    rgba = np.append(np.clip(rgb_color, 0.0, 1.0) * 255.0, 255.0).astype(np.uint8)
    material = trimesh.visual.material.SimpleMaterial(diffuse=rgba)
    mesh.visual = trimesh.visual.TextureVisuals(material=material)

    return mesh


def _rename_shared_mtl(obj_path, out_name):
    """trimesh 导出 .obj 时，配套材质文件固定命名为 material.mtl（与
    --out_name 无关）。如果同一个 --out_dir 下先后导出多个不同颜色的网格
    （如方形螺母 + 圆形螺母），后导出的会直接覆盖前一个的 material.mtl，
    导致先导出的网格颜色被"污染"。这里把本次导出的材质文件重命名为
    "<out_name>.mtl" 并同步更新 .obj 内的 mtllib 引用，让每个网格拥有
    独立的材质文件，互不覆盖。
    """
    obj_dir = os.path.dirname(obj_path)
    with open(obj_path, encoding="utf-8") as f:
        lines = f.readlines()

    mtl_name = None
    for line in lines:
        if line.startswith("mtllib "):
            mtl_name = line.split(maxsplit=1)[1].strip()
            break
    if mtl_name is None:
        return  # 没有引用材质文件，无需处理

    new_mtl_name = f"{out_name}.mtl"
    if mtl_name == new_mtl_name:
        return  # 已经是独立命名，无需重命名

    old_mtl_path = os.path.join(obj_dir, mtl_name)
    new_mtl_path = os.path.join(obj_dir, new_mtl_name)
    if os.path.isfile(old_mtl_path):
        os.replace(old_mtl_path, new_mtl_path)

    with open(obj_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"mtllib {new_mtl_name}\n" if line.startswith("mtllib ") else line)


def main():
    parser = argparse.ArgumentParser(
        description="将 square-nut.xml 转换为 FoundationPose 可用的 CAD 网格文件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xml",     default=XML_PATH, help="螺母 xml 路径（如 square-nut.xml / round-nut.xml）")
    parser.add_argument("--texture", default=TEXTURE_PATH, help="贴图路径（用于估算网格颜色）")
    parser.add_argument("--fallback_color", default="0.65 0.49 0.24",
                        help="贴图文件不存在时使用的默认漫反射颜色（'R G B'，0~1，空格分隔）。"
                             "默认是黄铜色近似值；如 round-nut.xml 引用的贴图在本仓库中缺失，"
                             "可传入 '0.75 0.76 0.78' 之类的钢灰色近似值。")
    parser.add_argument("--out_dir", default="nut_mesh", help="输出目录")
    parser.add_argument("--out_name", default="textured_simple", help="输出文件名（不含扩展名）")
    parser.add_argument("--format",  default="obj", choices=["obj", "ply", "stl"],
                        help="导出网格格式（.obj 会同时生成 .mtl 材质文件）")
    args = parser.parse_args()

    print(f"[解析] {args.xml}")
    boxes = parse_box_geoms(args.xml)
    print(f"  共解析到 {len(boxes)} 个 box geom")

    print(f"[取色] {args.texture}")
    fallback_rgb = [float(v) for v in args.fallback_color.split()]
    rgb_color = get_average_color(args.texture, fallback_rgb)
    print(f"  平均颜色 (RGB, 0~1): {rgb_color}")

    print("[构建网格]")
    mesh = build_mesh(boxes, rgb_color)
    print(f"  顶点数: {len(mesh.vertices)}  面片数: {len(mesh.faces)}")
    print(f"  包围盒 (min~max, 米): {mesh.bounds[0]} ~ {mesh.bounds[1]}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.out_name}.{args.format}")
    mesh.export(out_path)

    if args.format == "obj":
        _rename_shared_mtl(out_path, args.out_name)

    print(f"\n[完成] 已导出网格: {out_path}")
    print("  姿态估计示例命令：")
    print(f"  python foundationpose_api.py --mesh {out_path} \\")
    print(f"      --input pose_estimation_data --cam_K_file pose_estimation_data/cam_K.txt")


if __name__ == "__main__":
    main()
