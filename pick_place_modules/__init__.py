"""
pick_place_modules
-------------------
10_pick_and_place_nuts_mujoco.py 相关的算法/基础设施代码集中存放目录，
便于查找与两名同事分别独立开发：
  - sim_common.py      ：非算法基础设施（MuJoCo 读写原语、坐标变换、
                         标定文件加载、3D 可视化绘制工具）。
  - nut_perception.py  ：感知算法（NutPerceptionModule）。
  - grasp_planner.py   ：抓取规划控制算法（solve_ik_position +
                         MotionQueuePlanner）。
"""
