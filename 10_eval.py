"""10_eval.py
闭环评估：加载训练好的 BC 策略，在 MuJoCo 里跑 N 次抓取，统计成功率。

用法（服务器上）：
    python 10_eval.py --episodes 20 --exec-k 4 --device cuda
"""

import argparse
import os
from collections import deque

os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import mujoco
import numpy as np
import torch

from bc_model import BCNet

RECORD_EVERY = 25  # 20 Hz，和训练数据一致


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/home/lqs/robot-project/runs/bc/best.pt")
    p.add_argument("--xml", default="/home/lqs/robot-project/assets/franka_emika_panda/scene_pick.xml")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--exec-k", type=int, default=4, help="每次预测后执行几步")
    p.add_argument("--max-seconds", type=float, default=4.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    history = int(ckpt.get("history", 1))
    chunk = int(ckpt["chunk"])
    model = BCNet(chunk=chunk, qpos_dim=9 * history).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stats = ckpt["stats"]
    a_mean = np.array(stats["action_mean"], dtype=np.float32)
    a_std = np.array(stats["action_std"], dtype=np.float32)
    q_mean = np.array(stats["qpos_mean"], dtype=np.float32)
    q_std = np.array(stats["qpos_std"], dtype=np.float32)
    print(f"loaded {args.ckpt} (chunk={chunk}, history={history}, val={ckpt.get('val_loss', -1):.4f})")

    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    cube = m.body("cube").id
    hand = m.body("hand").id
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.55, 0.0, 0.45]
    cam.distance = 0.85
    cam.azimuth = 135
    cam.elevation = -30
    renderer = mujoco.Renderer(m, height=128, width=128)

    qbuf = deque(maxlen=history)

    def reset(cube_xy):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[9] = cube_xy[0]
        d.qpos[10] = cube_xy[1]
        mujoco.mj_forward(m, d)
        qbuf.clear()
        for _ in range(history):
            qbuf.append(d.qpos[:9].copy())

    def observe():
        qbuf.append(d.qpos[:9].copy())
        renderer.update_scene(d, camera=cam)
        img = renderer.render()
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        qs = np.stack(list(qbuf)).astype(np.float32)
        qs = (qs - q_mean) / q_std
        q = torch.from_numpy(qs.reshape(-1))
        return img.unsqueeze(0).to(device), q.unsqueeze(0).to(device)

    def act(img, q):
        with torch.no_grad():
            pred = model(img, q)[0].cpu().numpy()
        return pred * a_std + a_mean

    def run_episode(cube_xy):
        reset(cube_xy)
        max_z = 0.0
        steps = 0
        max_steps = int(args.max_seconds / m.opt.timestep)
        while steps < max_steps:
            img, q = observe()
            actions = act(img, q)
            if args.debug:
                a0 = actions[0]
                print(
                    f"  t={steps * m.opt.timestep:.2f}  "
                    f"hand={d.xpos[hand].round(3)}  cube={d.xpos[cube].round(3)}  "
                    f"pred_grip={a0[7]:6.1f}  pred_j1={a0[0]:+.3f}"
                )
            for k in range(min(args.exec_k, chunk)):
                a = actions[k]
                for _ in range(RECORD_EVERY):
                    d.ctrl[:7] = a[:7]
                    d.ctrl[7] = a[7]
                    mujoco.mj_step(m, d)
                    steps += 1
                max_z = max(max_z, float(d.xpos[cube][2]))
                if steps >= max_steps:
                    break
        return max_z > 0.5, max_z, steps

    results = []
    for i in range(args.episodes):
        x = float(np.random.uniform(0.45, 0.65))
        y = float(np.random.uniform(-0.10, 0.10))
        ok, max_z, steps = run_episode((x, y))
        results.append(ok)
        print(
            f"ep {i + 1:02d}  x={x:.3f} y={y:+.3f}  "
            f"成功={ok}  max_z={max_z:.3f}  steps={steps}"
        )

    rate = sum(results) / len(results)
    print(f"成功率: {sum(results)}/{len(results)} = {rate:.1%}")


if __name__ == "__main__":
    main()
