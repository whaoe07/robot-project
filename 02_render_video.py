"""02_render_video.py
把一段控制策略渲染成 mp4。不需要打开窗口，直接画到内存。
"""
import os
os.environ["MUJOCO_GL"] = "egl"   # 用离屏 EGL 渲染，不依赖窗口

import mujoco
import numpy as np
import imageio.v2 as imageio

XML = "/home/lqs/mujoco_menagerie/franka_emika_panda/scene.xml"
OUT = "/mnt/c/Users/lqs/Documents/deepseek-harness/default-workspace/franka_demo.mp4"

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=360, width=480)
writer = imageio.get_writer(OUT, fps=50)

STEPS = 500          # 1500 * 0.002s = 3 秒
RENDER_EVERY = 10     # 每 10 个物理步出一帧 -> 50 fps

for step in range(STEPS):
    t = step * model.opt.timestep
    # 手写策略：每个电机不同相位的正弦，让机械臂平滑摆动
    data.ctrl[:] = 1 * np.sin(2 * np.pi * 2 * t + np.arange(model.nu))
    mujoco.mj_step(model, data)
    if step % RENDER_EVERY == 0:
        renderer.update_scene(data)
        writer.append_data(renderer.render())

writer.close()
renderer.close()
print("saved", OUT)
