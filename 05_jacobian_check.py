import os
os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np

XML = "/home/lqs/mujoco_menagerie/franka_emika_panda/scene_pick.xml"
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

hand = m.body("hand").id
q0 = d.qpos[:7].copy()


def hand_pos(q):
    d.qpos[:7] = q
    mujoco.mj_forward(m, d)
    return d.xpos[hand].copy()


p0 = hand_pos(q0)

# 方法一：数值法。让每个关节单独动一丁点，看手移动多少
eps = 1e-6
J_num = np.zeros((3, 7))
for i in range(7):
    q = q0.copy()
    q[i] += eps
    J_num[:, i] = (hand_pos(q) - p0) / eps

# 方法二：MuJoCo 自带的解析雅可比
d.qpos[:7] = q0
mujoco.mj_forward(m, d)
jacp = np.zeros((3, m.nv))
jacr = np.zeros((3, m.nv))
mujoco.mj_jacBody(m, d, jacp, jacr, hand)
J_ana = jacp[:, :7]

print("两种算法最大差异:", np.abs(J_num - J_ana).max())
print("关节 1 那一列:", J_ana[:, 0].round(4))