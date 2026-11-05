"""08_record.py — 采集专家数据（6cm 方块版）

改动：
  - 方块 5cm -> 6cm，给对准更多余量
  - 专家速度 0.5 -> 0.3，帧数更多
  - 关键阶段加 0.3 秒停顿，让阶段更分明
  - 采 200 条，存到 data/raw6
"""

import os

os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import mujoco
import numpy as np

XML = "/home/lqs/mujoco_menagerie/franka_emika_panda/scene_pick.xml"
OUT = Path("/home/lqs/robot-project/data/raw6")
OUT.mkdir(parents=True, exist_ok=True)

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
hand = m.body("hand").id
cube = m.body("cube").id

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
R_home = d.xmat[hand].reshape(3, 3).copy()
q_home = d.qpos[:7].copy()

CAM = mujoco.MjvCamera()
CAM.lookat[:] = [0.55, 0.0, 0.45]
CAM.distance = 0.85
CAM.azimuth = 135
CAM.elevation = -30

renderer = mujoco.Renderer(m, height=128, width=128)

RECORD_EVERY = 25  # 20 Hz
GRASP_Z = 0.505
buf = None
step_i = 0


def solve_ik(target_p, q_start, damp=1e-3, iters=600, dt=0.6):
    d.qpos[:7] = q_start
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
        d.qpos[:7] += dt * dq
        d.qpos[:7] = np.clip(d.qpos[:7], m.jnt_range[:7, 0], m.jnt_range[:7, 1])
    mujoco.mj_forward(m, d)
    err = float(np.linalg.norm(target_p - d.xpos[hand]))
    return d.qpos[:7].copy(), err


def reset_scene(cube_xy):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[9] = cube_xy[0]
    d.qpos[10] = cube_xy[1]
    mujoco.mj_forward(m, d)


def start_episode():
    global buf, step_i
    buf = {"images": [], "qpos": [], "action": []}
    step_i = 0


def sim_step():
    global step_i
    mujoco.mj_step(m, d)
    step_i += 1
    if step_i % RECORD_EVERY == 0:
        renderer.update_scene(d, camera=CAM)
        buf["images"].append(renderer.render().copy())
        buf["qpos"].append(d.qpos[:9].copy())
        buf["action"].append(d.ctrl.copy())


def hold(seconds):
    n = int(seconds / m.opt.timestep)
    for _ in range(n):
        sim_step()


def move_to_and_wait(q_target, gripper, tol=0.02, speed=0.3, timeout=6.0):
    q_start = d.qpos[:7].copy()
    dist = float(np.max(np.abs(q_target - q_start)))
    duration = max(0.4, dist / speed)
    n = int(duration / m.opt.timestep)
    for i in range(n):
        a = (i + 1) / n
        d.ctrl[:7] = (1 - a) * q_start + a * q_target
        d.ctrl[7] = gripper
        sim_step()
    t = 0.0
    while t < timeout:
        d.ctrl[:7] = q_target
        d.ctrl[7] = gripper
        sim_step()
        t += m.opt.timestep
        if np.abs(d.qpos[:7] - q_target).max() < tol:
            return True
    return False


def gripper_sweep(target, seconds=0.8):
    start = d.ctrl[7]
    n = int(seconds / m.opt.timestep)
    for i in range(n):
        d.ctrl[7] = start + (target - start) * (i + 1) / n
        sim_step()


def run_episode(cube_xy):
    reset_scene(cube_xy)
    q_pre, e1 = solve_ik(np.array([cube_xy[0], cube_xy[1], 0.56]), q_home)
    q_grasp, e2 = solve_ik(np.array([cube_xy[0], cube_xy[1], GRASP_Z]), q_pre)
    q_lift, e3 = solve_ik(np.array([cube_xy[0], cube_xy[1], 0.63]), q_grasp)
    ik_ok = max(e1, e2, e3) < 0.005
    if not ik_ok:
        return None, False, False

    reset_scene(cube_xy)
    start_episode()

    move_to_and_wait(q_pre, 255)
    hold(0.3)
    move_to_and_wait(q_grasp, 255)
    hold(0.3)
    gripper_sweep(0.0, 0.8)
    hold(0.3)
    move_to_and_wait(q_lift, 0.0)
    hold(0.3)

    success = float(d.xpos[cube][2]) > 0.5
    return buf, success, True


def save_episode(i, buf, success, cube_xy):
    path = OUT / f"ep_{i:04d}.npz"
    np.savez_compressed(
        path,
        images=np.stack(buf["images"]),
        qpos=np.stack(buf["qpos"]),
        action=np.stack(buf["action"]),
        success=np.array(success),
        cube_xy=np.array(cube_xy, dtype=np.float32),
    )
    return path


if __name__ == "__main__":
    N = 200
    total_ok = 0
    for i in range(N):
        x = float(np.random.uniform(0.45, 0.65))
        y = float(np.random.uniform(-0.10, 0.10))
        buf, success, ik_ok = run_episode((x, y))
        if buf is None:
            print(f"ep {i:02d}  x={x:.3f} y={y:+.3f}  IK 失败，跳过")
            continue
        path = save_episode(i, buf, success, (x, y))
        total_ok += int(success)
        print(
            f"ep {i:02d}  x={x:.3f} y={y:+.3f}  "
            f"帧数={len(buf['images'])}  成功={success}  -> {path.name}"
        )
    print(f"完成: {total_ok}/{N} 成功")
