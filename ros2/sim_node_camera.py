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
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

XML = os.path.expanduser("~/mujoco_menagerie/franka_emika_panda/scene_pick.xml")
IMG_H, IMG_W = 128, 128


def make_scene_with_size(size_cm):
    """按方块边长生成一个临时 XML。

    为什么不直接改 model.geom_size：
      MuJoCo 的几何尺寸是编译期参数，运行时改会让惯性/碰撞不一致，
      方块会被挤飞（实测 max_z 出现 1000+ 的异常值）。
    """
    half = size_cm / 200.0                 # cm → m → 半边长
    mass = 0.05 * (size_cm / 5.0) ** 3     # 体积等比缩放
    z = 0.4 + half                         # 贴桌面

    src = Path(XML).read_text(encoding="utf-8")
    dst = src.replace(
        'size="0.025 0.025 0.025"',
        f'size="{half:.6f} {half:.6f} {half:.6f}"',
    ).replace(
        'mass="0.05"',
        f'mass="{mass:.6f}"',
    ).replace(
        '<body name="cube" pos="0.55 0 0.43">',
        f'<body name="cube" pos="0.55 0 {z:.6f}">',
    )

    out_path = Path(XML).parent / f"scene_pick_s{size_cm:g}.xml"
    out_path.write_text(dst, encoding="utf-8")
    return str(out_path), half, mass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hz", type=float, default=20.0)
    p.add_argument("--log-every", type=int, default=20,
                   help="每多少个 tick 打印一次日志（20Hz 下 20 = 1 秒）")
    p.add_argument("--cmd-topic", default="/joint_command")
    p.add_argument("--cube-x", type=float, default=None, help="方块 x（默认 0.55）")
    p.add_argument("--cube-y", type=float, default=None, help="方块 y（默认 0）")
    p.add_argument("--cube-vel-x", type=float, default=0.0,
                   help="方块 x 方向速度 m/s（模拟传送带）")
    p.add_argument("--cube-vel-y", type=float, default=0.0,
                   help="方块 y 方向速度 m/s")
    p.add_argument("--cube-size", type=float, default=None,
                   help="方块边长（厘米），默认用 XML 里的 5cm")
    return p.parse_args()


class SimNodeCamera(Node):
    def __init__(self, args):
        super().__init__("sim_node")

        # ---------- MuJoCo ----------
        self.cube_half = 0.025
        xml_path = XML
        if args.cube_size is not None:
            xml_path, self.cube_half, mass = make_scene_with_size(args.cube_size)
            self.get_logger().info(
                f"cube size {args.cube_size}cm -> {xml_path} "
                f"(half={self.cube_half:.4f}, mass={mass:.4f}kg)"
            )

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # 可选：指定方块位置（qpos[9]=x, qpos[10]=y）
        if args.cube_x is not None:
            self.data.qpos[9] = args.cube_x
        if args.cube_y is not None:
            self.data.qpos[10] = args.cube_y
        mujoco.mj_forward(self.model, self.data)

        self.hand_id = self.model.body("hand").id
        self.cube_id = self.model.body("cube").id

        self.hz = args.hz
        self.steps_per_tick = int(round((1.0 / self.hz) / self.model.opt.timestep))

        # 方块速度（传送带）：每物理步强制设置，抵消摩擦减速
        self.cube_vel_x = args.cube_vel_x
        self.cube_vel_y = args.cube_vel_y
        self.conveyor = (self.cube_vel_x != 0.0 or self.cube_vel_y != 0.0)

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
        self.log_every = args.log_every

    def on_command(self, msg):
        n = min(len(msg.position), self.model.nu)
        self.data.ctrl[:n] = msg.position[:n]
        self.cmd_count += 1

    def tick(self):
        # 1. 推进物理（传送带模式下每步强制设置方块速度）
        for _ in range(self.steps_per_tick):
            if self.conveyor:
                self.data.qvel[9] = self.cube_vel_x   # 线速度 x
                self.data.qvel[10] = self.cube_vel_y  # 线速度 y
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
        if self.tick_count % self.log_every == 0:
            hand = self.data.xpos[self.hand_id]
            cube = self.data.xpos[self.cube_id]
            self.get_logger().info(
                f"t={self.data.time:.1f}s hand_z={hand[2]:.2f} "
                f"cube_x={cube[0]:.3f} cube_y={cube[1]:+.3f} "
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
