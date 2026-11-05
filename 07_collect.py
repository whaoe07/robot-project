import os

os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np

XML = "/home/lqs/mujoco_menagerie/franka_emika_panda/scene_pick.xml"
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
hand = m.body("hand").id
cube = m.body("cube").id

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
R_home = d.xmat[hand].reshape(3, 3).copy()
q_home = d.qpos[:7].copy()


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

def reset_scene(cube_xy=(0.55, 0.0)):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[9] = cube_xy[0]
    d.qpos[10] = cube_xy[1]
    mujoco.mj_forward(m, d)


def move_to_and_wait(q_target, gripper, tol=0.02, speed=1.0, timeout=4.0):
    q_start = d.qpos[:7].copy()
    dist = float(np.max(np.abs(q_target - q_start)))
    duration = max(0.3, dist / speed)
    n = int(duration / m.opt.timestep)
    for i in range(n):
        a = (i + 1) / n
        d.ctrl[:7] = (1 - a) * q_start + a * q_target
        d.ctrl[7] = gripper
        mujoco.mj_step(m, d)
    t = 0.0
    while t < timeout:
        d.ctrl[:7] = q_target
        d.ctrl[7] = gripper
        mujoco.mj_step(m, d)
        t += m.opt.timestep
        if np.abs(d.qpos[:7] - q_target).max() < tol:
            return True, float(t)
    return False, float(timeout)


def gripper_sweep(target, seconds=0.8):
    start = d.ctrl[7]
    n = int(seconds / m.opt.timestep)
    for i in range(n):
        d.ctrl[7] = start + (target - start) * (i + 1) / n
        mujoco.mj_step(m, d)


def run_episode(cube_xy=(0.55, 0.0)):
    reset_scene(cube_xy)
    q_pre,e1 = solve_ik(np.array([cube_xy[0], cube_xy[1], 0.56]), q_home)
    q_grasp,e2 = solve_ik(np.array([cube_xy[0], cube_xy[1], 0.505]), q_pre)
    q_lift,e3 = solve_ik(np.array([cube_xy[0], cube_xy[1], 0.63]), q_grasp)
    ik_ok = max(e1, e2, e3) < 0.005
    if not ik_ok:
      return {
        "ik_ok": False, "lifted": False, "cube_z": float(d.xpos[cube][2]),
        "ok_pre": False, "ok_grasp": False, "ok_lift": False, "t": 0.0,
      }
    reset_scene(cube_xy)

    ok_pre, t1 = move_to_and_wait(q_pre, 255)
    ok_grasp, t2 = move_to_and_wait(q_grasp, 255)
    gripper_sweep(0.0, 0.8)
    ok_lift, t3 = move_to_and_wait(q_lift, 0.0)
    cube_z = float(d.xpos[cube][2])
    return {
        "ik_ok": ik_ok,
        "lifted": cube_z > 0.5,
        "cube_z": cube_z,
        "ok_pre": ok_pre,
        "ok_grasp": ok_grasp,
        "ok_lift": ok_lift,
        "t": t1 + t2 + t3,
    }


if __name__ == "__main__":
    results = []
    for k in range(20):
        x = np.random.uniform(0.45, 0.65)
        y = np.random.uniform(-0.10, 0.10)
        r = run_episode(cube_xy=(x, y))
        results.append(r)
        print(
            f"第{k+1:2d}次  x={x:.3f} y={y:+.3f}  "
            f"ik={r['ik_ok']}  抓起={r['lifted']}  z={r['cube_z']:.3f}"
        )
    ok = sum(1 for r in results if r["lifted"] and r["ik_ok"])
    print(f"成功率: {ok}/{len(results)}")