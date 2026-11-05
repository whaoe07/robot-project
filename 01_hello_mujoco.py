"""01_hello_mujoco.py
第一个 MuJoCo 脚本：加载模型，跑一个随机策略。
目的：亲眼看到"状态 -> 控制 -> 新状态"这个循环。
"""

import mujoco
import numpy as np

# 模型文件路径。~ 在 Python 里不会自动展开，所以写绝对路径
MODEL_PATH = "/home/lqs/mujoco_menagerie/franka_emika_panda/scene.xml"

# mjModel：静态模型。关节、连杆、质量、电机都在这里，加载后不再变
model = mujoco.MjModel.from_xml_path(MODEL_PATH)

# mjData：运行时状态。位置、速度、受力，每个时间步都在变
data = mujoco.MjData(model)

# 先算一次派生量（正运动学、碰撞检测），让 data 从一致状态开始
mujoco.mj_forward(model, data)

print("nq =", model.nq, "  # 位置自由度（关节角）个数")
print("nv =", model.nv, "  # 速度自由度个数")
print("nu =", model.nu, "  # 执行器（电机）个数")
print("timestep =", model.opt.timestep, "秒")

# 控制循环：所有机器人算法的骨架
for step in range(1000):
    # 1) 策略：先随机代替，以后换成神经网络
    data.ctrl[:] = np.random.uniform(-1.0, 1.0, model.nu)

    # 2) 物理引擎：按当前控制量，把世界推进一个 timestep
    mujoco.mj_step(model, data)

# 1000 步 × 0.002 秒 = 2 秒仿真时间
print("仿真结束，前 7 个关节角：")
print(data.qpos[:7])
