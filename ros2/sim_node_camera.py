"""sim_node_camera.py — 在 sim_node_mujoco 基础上，增加相机图像发布。

新增：
  - 用 MuJoCo Renderer 离屏渲染 128x128 图像
  - 以 sensor_msgs/Image（rgb8）发到 /camera/image

运行：
    source /opt/ros/humble/setup.bash
    python3 ros2/sim_node_camera.py
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

XML = os.path.expanduser("~/mujoco_menagerie/franka_emika_panda/scene_pick.xml")

CTRL_HZ = 10.0
IMG_H, IMG_W = 128, 128


class SimNodeCamera(Node):
    def __init__(self):
        super().__init__("sim_node")

        # ---------- MuJoCo ----------
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.hand_id = self.model.body("hand").id
        self.cube_id = self.model.body("cube").id

        self.steps_per_tick = int(round((1.0 / CTRL_HZ) / self.model.opt.timestep))

        # ---------- 相机 ----------
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.55, 0.0, 0.45]
        self.cam.distance = 0.85
        self.cam.azimuth = 135
        self.cam.elevation = -30

        self.renderer = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)
        self.get_logger().info(
            f"MuJoCo loaded: steps_per_tick={self.steps_per_tick}, "
            f"image={IMG_W}x{IMG_H} rgb8"
        )

        # ---------- ROS2 ----------
        self.pub_joint = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_img = self.create_publisher(Image, "/camera/image", 10)
        self.sub = self.create_subscription(
            JointState, "/joint_command", self.on_command, 10
        )
        self.timer = self.create_timer(1.0 / CTRL_HZ, self.tick)

        self.tick_count = 0
        self.cmd_count = 0
        self.img_count = 0

    def on_command(self, msg):
        n = min(len(msg.position), self.model.nu)
        self.data.ctrl[:n] = msg.position[:n]
        self.cmd_count += 1

    def tick(self):
        # 1. 推进物理
        for _ in range(self.steps_per_tick):
            mujoco.mj_step(self.model, self.data)

        stamp = self.get_clock().now().to_msg()

        # 2. 发布关节状态
        js = JointState()
        js.header.stamp = stamp
        js.header.frame_id = "world"
        js.name = [f"q{i}" for i in range(self.model.nq)]
        js.position = [float(v) for v in self.data.qpos[: self.model.nq]]
        js.velocity = [float(v) for v in self.data.qvel[: self.model.nv]]
        self.pub_joint.publish(js)

        # 3. 渲染并发布图像
        self.renderer.update_scene(self.data, camera=self.cam)
        img = self.renderer.render()  # (H, W, 3) uint8

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"
        msg.height = IMG_H
        msg.width = IMG_W
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = IMG_W * 3  # 每行字节数 = 宽 * 3 通道
        msg.data = img.tobytes()
        self.pub_img.publish(msg)
        self.img_count += 1

        # 4. 日志
        self.tick_count += 1
        if self.tick_count % 20 == 0:
            hand = self.data.xpos[self.hand_id]
            cube = self.data.xpos[self.cube_id]
            self.get_logger().info(
                f"t={self.data.time:.1f}s hand_z={hand[2]:.2f} "
                f"cube_z={cube[2]:.3f} imgs={self.img_count} cmds={self.cmd_count}"
            )

    def destroy_node(self):
        self.renderer.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = SimNodeCamera()
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
