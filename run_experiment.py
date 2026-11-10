"""run_experiment.py — 批量跑时延实验

对每组 (延迟, 抖动, 丢包) 跑 N 次：
  启动 sim_node + policy_node + latency_node
  等固定时长
  从日志里取最终 cube_z
  杀掉三个节点，写 CSV

用法：
    python3 run_experiment.py           # 默认每组 3 次
    python3 run_experiment.py 5         # 每组 5 次
    DURATION=10 python3 run_experiment.py 1   # 快速验证

输出：
    results/exp_<时间戳>.csv
    results/exp_<时间戳>_raw.txt
"""

import csv
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJ = Path.home() / "robot-project"
RESULT_DIR = PROJ / "results"

# (延迟ms, 抖动ms, 丢包率, 方块速度 m/s, 方块边长 cm)
GROUPS = [
    (0, 0, 0.0, 0.003, 4.0),
    (0, 0, 0.0, 0.003, 5.0),   # 训练分布内
    (0, 0, 0.0, 0.003, 6.0),
    (0, 0, 0.0, 0.003, 7.0),
]

CUBE_START_X = 0.55   # 起点固定，在训练分布中心

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
DURATION = float(os.environ.get("DURATION", "40"))
SETTLE_SIM = 3.0     # 等仿真起来
SETTLE_POLICY = 2.0  # 等策略加载完

CUBE_Z_RE = re.compile(r"cube_z=([0-9.]+)")


def kill_all():
    subprocess.run(
        ["bash", "-c",
         "pkill -f 'ros2/sim_node_camera.py' ; "
         "pkill -f 'ros2/policy_node.py' ; "
         "pkill -f latency_node ; true"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)


def spawn(bash_cmd, log_fp):
    """在独立进程组里启动，方便整组杀掉。"""
    return subprocess.Popen(
        ["bash", "-c", bash_cmd],
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def terminate(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULT_DIR / f"exp_{stamp}.csv"
    raw_path = RESULT_DIR / f"exp_{stamp}_raw.txt"

    print(f"=== 实验开始 {datetime.now()} ===")
    print(f"每组 {TRIALS} 次，每次 {DURATION}s，共 {len(GROUPS)} 组")
    print(f"结果: {csv_path}")
    print()
    print("实验组：")
    for d, j, p, v, s in GROUPS:
        print(f"  延迟={d}ms 抖动={j}ms 丢包={p} 速度={v}m/s 边长={s}cm")
    print()

    rows = []
    total_ok = 0
    total_run = 0

    raw_fp = open(raw_path, "w", encoding="utf-8", buffering=1)

    try:
        for delay, jitter, drop, vel, size in GROUPS:
            for trial in range(1, TRIALS + 1):
                print(f"--- delay={delay} jitter={jitter} drop={drop} "
                      f"vel={vel} size={size}cm trial={trial}/{TRIALS} ---",
                      flush=True)

                kill_all()

                raw_fp.write(
                    f"\n=== delay={delay} jitter={jitter} drop={drop} "
                    f"vel={vel} size={size} trial={trial} ===\n")
                raw_fp.flush()
                start_pos = raw_fp.tell()

                sim = policy = lat = None
                try:
                    sim = spawn(
                        f"source /opt/ros/humble/setup.bash && "
                        f"cd {PROJ} && exec python3 ros2/sim_node_camera.py "
                        f"--cmd-topic /joint_command_delayed "
                        f"--cube-x {CUBE_START_X} --cube-vel-x {vel} "
                        f"--cube-size {size}",
                        raw_fp,
                    )
                    time.sleep(SETTLE_SIM)

                    policy = spawn(
                        f"source /opt/ros/humble/setup.bash && "
                        f"cd {PROJ} && exec python3 ros2/policy_node.py",
                        raw_fp,
                    )
                    time.sleep(SETTLE_POLICY)

                    lat = spawn(
                        f"source /opt/ros/humble/setup.bash && "
                        f"source {PROJ}/ros2_ws/install/setup.bash && "
                        f"exec ros2 run robot_cpp latency_node --ros-args "
                        f"-p delay_ms:={delay} -p jitter_ms:={jitter} "
                        f"-p drop_rate:={drop}",
                        raw_fp,
                    )

                    time.sleep(DURATION)
                finally:
                    terminate(lat)
                    terminate(policy)
                    terminate(sim)
                    raw_fp.flush()

                # 解析本次日志段
                raw_fp.flush()
                with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(start_pos)
                    segment = f.read()

                zs = [float(m) for m in CUBE_Z_RE.findall(segment)]
                max_z = max(zs) if zs else 0.0
                final_z = zs[-1] if zs else 0.0

                # 成功判据：抬离桌面 4cm（随方块尺寸变化）
                init_z = 0.4 + (size / 200.0)
                thresh = init_z + 0.04
                success = 1 if final_z > thresh else 0

                # 抓住之后的最低高度：检测"抬起来又掉下去"
                grasp_idx = next((i for i, z in enumerate(zs) if z > thresh), None)
                min_z_after = min(zs[grasp_idx:]) if grasp_idx is not None else final_z

                print(f"  结果: success={success} final_z={final_z:.3f} "
                      f"max_z={max_z:.3f} min_z_after={min_z_after:.3f} "
                      f"阈值={thresh:.3f} (样本{len(zs)}个)", flush=True)

                rows.append({
                    "delay_ms": delay,
                    "jitter_ms": jitter,
                    "drop_rate": drop,
                    "cube_vel_x": vel,
                    "cube_size_cm": size,
                    "trial": trial,
                    "success": success,
                    "final_cube_z": round(final_z, 4),
                    "max_cube_z": round(max_z, 4),
                    "min_z_after_grasp": round(min_z_after, 4),
                })
                total_ok += success
                total_run += 1

                kill_all()
    finally:
        raw_fp.close()
        kill_all()

    # 写 CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "delay_ms", "jitter_ms", "drop_rate", "cube_vel_x",
            "cube_size_cm", "trial",
            "success", "final_cube_z", "max_cube_z", "min_z_after_grasp"])
        writer.writeheader()
        writer.writerows(rows)

    # 汇总
    print()
    print("========== 汇总 ==========")
    stats = {}
    for r in rows:
        key = r["cube_size_cm"]
        s = stats.setdefault(key, {"n": 0, "ok": 0, "z": 0.0, "minz": 0.0})
        s["n"] += 1
        s["ok"] += r["success"]
        s["z"] += r["final_cube_z"]
        s["minz"] += r["min_z_after_grasp"]

    print(f"{'边长cm':<10}{'次数':<8}{'成功率':<10}"
          f"{'平均最终高度':<16}{'抓后最低'}")
    for key in sorted(stats):
        s = stats[key]
        print(f"{key:<10}{s['n']:<8}{s['ok'] / s['n'] * 100:<10.0f}"
              f"{s['z'] / s['n']:<16.3f}{s['minz'] / s['n']:.3f}")

    print()
    print(f"总成功率: {total_ok}/{total_run}")
    print(f"详细数据: {csv_path}")


if __name__ == "__main__":
    main()
