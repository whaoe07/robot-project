"""sim_node_mujoco.py — ROS2 仿真节点（接真实 MuJoCo）。

职责：
  1. 加载 MuJoCo 模型，持续推进物理
  2. 每 0.1 秒（10 Hz）发布一次 /joint_states（真实关节角度）
  3. 订阅命令话题（默认 /joint_command，可用 --cmd-topic 改），把目标角写进 ctrl

时间关系（重要）：
  - MuJoCo 物理步长 = 0.002 秒
  - ROS2 定时器   = 0.1 秒
  - 所以每次定时器要补 50 个物理步：0.1 / 0.002 = 50

运行：
    # 直接接收命令
    python3 ros2/sim_node_mujoco.py

    # 通过时延节点
    python3 ros2/sim_node_mujoco.py --cmd-topic /joint_command_delayed
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

XML = os.path.expanduser("~/mujoco_menagerie/franka_emika_panda/scene_pick.xml")

CTRL_HZ = 10.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cmd-topic", default="/joint_command")
    return p.parse_args()


class SimNode(Node):
    def __init__(self, cmd_topic):
        super().__init__("sim_node")

        # ---------- MuJoCo ----------
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.hand_id = self.model.body("hand").id
        self.cube_id = self.model.body("cube").id

        self.steps_per_tick = int(round(
            (1.0 / CTRL_HZ) / self.model.opt.timestep
        ))
        self.get_logger().info(
            f"MuJoCo loaded: dt={self.model.opt.timestep}s, "
            f"steps_per_tick={self.steps_per_tick}, cmd_topic={cmd_topic}"
        )

        # ---------- ROS2 ----------
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.cmd_topic = cmd_topic
        self.sub = self.create_subscription(
            JointState, cmd_topic, self.on_command, 10
        )
        self.timer = self.create_timer(1.0 / CTRL_HZ, self.tick)

        self.tick_count = 0
        self.cmd_count = 0

    def on_command(self, msg):
        n = min(len(msg.position), self.model.nu)
        self.data.ctrl[:n] = msg.position[:n]
        self.cmd_count += 1

    def tick(self):
        for _ in range(self.steps_per_tick):
            mujoco.mj_step(self.model, self.data)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"

        nq = self.model.nq
        msg.name = [f"q{i}" for i in range(nq)]
        msg.position = [float(v) for v in self.data.qpos[:nq]]
        msg.velocity = [float(v) for v in self.data.qvel[: self.model.nv]]
        msg.effort = []

        self.pub.publish(msg)
        self.tick_count += 1

        if self.tick_count % 20 == 0:
            t = self.data.time
            hand = self.data.xpos[self.hand_id]
            cube = self.data.xpos[self.cube_id]
            self.get_logger().info(
                f"t={t:.1f}s hand=({hand[0]:.2f},{hand[1]:.2f},{hand[2]:.2f}) "
                f"cube_z={cube[2]:.3f} cmds={self.cmd_count}"
            )


def main():
    args = parse_args()
    rclpy.init()
    node = SimNode(args.cmd_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
