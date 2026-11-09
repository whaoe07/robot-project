"""sim_node_camera.py — 仿真节点：发布关节状态 + 相机图像，接收命令

职责：
  1. 加载 MuJoCo，推进物理
  2. 以 --hz 频率（默认 20）发布 /joint_states 和 /camera/image
  3. 订阅命令话题（默认 /joint_command），写进 ctrl

时间关系（重要）：
  - 物理步长 0.002 秒
  - 控制周期 1/hz 秒
  - steps_per_tick = (1/hz) / 0.002
    20 Hz → 25 步；10 Hz → 50 步

运行：
    # 直接接收命令（无延迟）
    python3 ros2/sim_node_camera.py

    # 经过时延节点
    python3 ros2/sim_node_camera.py --cmd-topic /joint_command_delayed
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

XML = os.path.expanduser("~/mujoco_menagerie/franka_emika_panda/scene_pick.xml")
IMG_H, IMG_W = 128, 128


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hz", type=float, default=20.0)
    p.add_argument("--cmd-topic", default="/joint_command")
    p.add_argument("--cube-x", type=float, default=None, help="方块 x（默认 0.55）")
    p.add_argument("--cube-y", type=float, default=None, help="方块 y（默认 0）")
    return p.parse_args()


class SimNodeCamera(Node):
    def __init__(self, args):
        super().__init__("sim_node")

        # ---------- MuJoCo ----------
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # 可选：指定方块位置（qpos[9], qpos[10]）
        if args.cube_x is not None:
            self.data.qpos[9] = args.cube_x
        if args.cube_y is not None:
            self.data.qpos[10] = args.cube_y
        mujoco.mj_forward(self.model, self.data)

        self.hand_id = self.model.body("hand").id
        self.cube_id = self.model.body("cube").id

        self.hz = args.hz
        self.steps_per_tick = int(round((1.0 / self.hz) / self.model.opt.timestep))

        # ---------- 相机 ----------
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.55, 0.0, 0.45]
        self.cam.distance = 0.85
        self.cam.azimuth = 135
        self.cam.elevation = -30
        self.renderer = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)

        self.get_logger().info(
            f"MuJoCo loaded: {self.hz}Hz, steps_per_tick={self.steps_per_tick}, "
            f"cmd_topic={args.cmd_topic}, cube=({self.data.qpos[9]:.3f},"
            f"{self.data.qpos[10]:.3f})"
        )

        # ---------- ROS2 ----------
        self.pub_joint = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_img = self.create_publisher(Image, "/camera/image", 10)
        self.sub = self.create_subscription(
            JointState, args.cmd_topic, self.on_command, 10
        )
        self.timer = self.create_timer(1.0 / self.hz, self.tick)

        self.tick_count = 0
        self.cmd_count = 0

    def on_command(self, msg):
        n = min(len(msg.position), self.model.nu)
        self.data.ctrl[:n] = msg.position[:n]
        self.cmd_count += 1

    def tick(self):
        # 1. 推进物理
        for _ in range(self.steps_per_tick):
            mujoco.mj_step(self.model, self.data)

        stamp = self.get_clock().now().to_msg()

        # 2. 关节状态
        js = JointState()
        js.header.stamp = stamp
        js.header.frame_id = "world"
        js.name = [f"q{i}" for i in range(self.model.nq)]
        js.position = [float(v) for v in self.data.qpos[: self.model.nq]]
        js.velocity = [float(v) for v in self.data.qvel[: self.model.nv]]
        self.pub_joint.publish(js)

        # 3. 图像
        self.renderer.update_scene(self.data, camera=self.cam)
        img = self.renderer.render()
        im = Image()
        im.header.stamp = stamp
        im.header.frame_id = "camera"
        im.height = IMG_H
        im.width = IMG_W
        im.encoding = "rgb8"
        im.is_bigendian = 0
        im.step = IMG_W * 3
        im.data = img.tobytes()
        self.pub_img.publish(im)

        # 4. 日志
        self.tick_count += 1
        if self.tick_count % 40 == 0:
            hand = self.data.xpos[self.hand_id]
            cube = self.data.xpos[self.cube_id]
            self.get_logger().info(
                f"t={self.data.time:.1f}s hand_z={hand[2]:.2f} "
                f"cube_z={cube[2]:.3f} cmds={self.cmd_count}"
            )

    def destroy_node(self):
        self.renderer.close()
        super().destroy_node()


def main():
    args = parse_args()
    rclpy.init()
    node = SimNodeCamera(args)
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
