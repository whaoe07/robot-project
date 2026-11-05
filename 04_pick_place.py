import os

os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
import imageio.v2 as imageio

XML = "/home/lqs/mujoco_menagerie/franka_emika_panda/scene_pick.xml"
OUT = "/home/lqs/robot-project/pick_demo.mp4"

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

hand = m.body("hand").id
cube = m.body("cube").id
R_home = d.xmat[hand].reshape(3, 3).copy()
q_home = d.qpos[:7].copy()


def solve_ik(target_p, q_start, iters=600, dt=0.6, damp=1e-3):
    d.qpos[:7] = q_start
    e_r = np.zeros(3)
    for _ in range(iters):
        mujoco.mj_forward(m, d)
        e_p = target_p - d.xpos[hand]
        R = d.xmat[hand].reshape(3, 3)
        e_r = 0.5 * (
            np.cross(R[:, 0], R_home[:, 0])
            + np.cross(R[:, 1], R_home[:, 1])
            + np.cross(R[:, 2], R_home[:, 2])
        )
        if np.linalg.norm(e_p) < 1e-3 and np.linalg.norm(e_r) < 1e-2:
            break
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, jacr, hand)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])
        e = np.concatenate([e_p, e_r])
        dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e)
        if _ % 100 == 0:
            print(_, "误差", round(float(np.linalg.norm(e_p)), 4), "|dq|",
    round(float(np.linalg.norm(dq)), 4))
        d.qpos[:7] += dt * dq
        d.qpos[:7] = np.clip(d.qpos[:7], m.jnt_range[:7, 0], m.jnt_range[:7, 1])
    mujoco.mj_forward(m, d)
    return d.qpos[:7].copy()

def move_until_reached(q_target, gripper, tol=0.02, timeout=3.0):
    """移动到 q_target，直到误差小于 tol 或超时。
    返回 True 表示到达，False 表示超时。"""
    t = 0.0
    while t < timeout:
        d.ctrl[:7] = q_target
        d.ctrl[7] = gripper
        mujoco.mj_step(m, d)
        t += m.opt.timestep
        err = np.abs(d.qpos[:7] - q_target).max()
        if err < tol:
            return True
    return False
    
q_pre = solve_ik(np.array([0.55, 0.0, 0.56]), q_home)
q_grasp = solve_ik(np.array([0.55, 0.0, 0.505]), q_pre)
q_lift = solve_ik(np.array([0.55, 0.0, 0.63]), q_grasp)
q_place = solve_ik(np.array([0.38, 0.18, 0.63]), q_lift)

print("ik ok")

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

renderer = mujoco.Renderer(m, height=360, width=480)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.5, 0.0, 0.42]
cam.distance = 1.5
cam.azimuth = 140
cam.elevation = -22
writer = imageio.get_writer(OUT, fps=50)

frame_count = 0


def step_and_maybe_render(render):
    global frame_count
    mujoco.mj_step(m, d)
    frame_count += 1
    if render and frame_count % 10 == 0:
        renderer.update_scene(d, camera=cam)
        writer.append_data(renderer.render())


def hold(steps):
    for _ in range(steps):
        step_and_maybe_render(True)


def move_to(q_target, seconds, gripper=None):
    q_start = d.qpos[:7].copy()
    n = int(seconds / m.opt.timestep)
    for i in range(n):
        a = (i + 1) / n
        d.ctrl[:7] = (1 - a) * q_start + a * q_target
        if gripper is not None:
            d.ctrl[7] = gripper
        step_and_maybe_render(True)


def gripper_to(value, seconds):
    g0 = d.ctrl[7]
    n = int(seconds / m.opt.timestep)
    for i in range(n):
        d.ctrl[7] = g0 + (value - g0) * (i + 1) / n
        step_and_maybe_render(True)


# 0. 初始稳定
d.ctrl[:7] = q_home
d.ctrl[7] = 255
hold(250)

# 1. 移到抓取点上方
move_to(q_pre, 1.5, gripper=255)
hold(150)

ok = move_until_reached(q_grasp, 255)
if not ok:
    print("没到位，跳过这次")
else:
    gripper_to(0.0, 0.8)

# 3. 闭合夹爪
gripper_to(0.0, 0.8)
hold(200)

# 5. 移到放置点
move_to(q_place, 1.5, gripper=0.0)
hold(150)

# 6. 松开
gripper_to(255.0, 0.6)
hold(150)

writer.close()
renderer.close()

print("cube z after pick:", round(float(d.xpos[cube][2]), 4))
print("frames:", frame_count, "saved:", OUT)

