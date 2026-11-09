"""policy_node.py — ACT 策略节点

职责：
  订阅 /camera/image + /joint_states
  维护最近 4 帧历史
  每 8 个控制周期重推理一次，输出 16 步动作，执行前 8 步
  发布 /joint_command

运行：
    source /opt/ros/humble/setup.bash
    cd ~/robot-project
    python3 ros2/policy_node.py
"""

import argparse
import os
import sys
from collections import deque

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

# 让 python 找到项目根目录下的 act_model.py
sys.path.insert(0, os.path.expanduser("~/robot-project"))
from act_model import ACTNet  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.expanduser("~/robot-project/models/best.pt"))
    p.add_argument("--hz", type=float, default=20.0)
    p.add_argument("--exec-k", type=int, default=8)
    p.add_argument("--out-topic", default="/joint_command")
    return p.parse_args()


class PolicyNode(Node):
    def __init__(self, args):
        super().__init__("policy_node")

        # ---------- 加载模型 ----------
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        self.chunk = int(ckpt["chunk"])
        self.history = int(ckpt.get("history", 4))
        self.model = ACTNet(chunk=self.chunk, history=self.history)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        s = ckpt["stats"]
        self.a_mean = np.array(s["action_mean"], dtype=np.float32)
        self.a_std = np.array(s["action_std"], dtype=np.float32)
        self.q_mean = np.array(s["qpos_mean"], dtype=np.float32)
        self.q_std = np.array(s["qpos_std"], dtype=np.float32)
        self.get_logger().info(
            f"ACT loaded: chunk={self.chunk}, history={self.history}, "
            f"val={ckpt.get('val_loss', -1):.4f}"
        )

        # ---------- 历史缓冲 ----------
        self.img_buf = deque(maxlen=self.history)
        self.q_buf = deque(maxlen=self.history)
        self.latest_img = None
        self.latest_q = None

        # ---------- 动作块 ----------
        self.chunk_actions = None  # (chunk, 8) 未归一化的动作
        self.chunk_idx = 0
        self.exec_k = args.exec_k

        # ---------- ROS2 ----------
        self.sub_img = self.create_subscription(
            Image, "/camera/image", self.on_image, 10
        )
        self.sub_q = self.create_subscription(
            JointState, "/joint_states", self.on_joint, 10
        )
        self.pub = self.create_publisher(JointState, args.out_topic, 10)
        self.timer = self.create_timer(1.0 / args.hz, self.tick)

        self.infer_count = 0
        self.pub_count = 0
        self.get_logger().info(f"policy_node: {args.hz} Hz, exec_k={args.exec_k}")

    def on_image(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
        self.latest_img = img.copy()

    def on_joint(self, msg):
        # 只取前 9 个（7 个手臂 + 2 个手指）
        q = np.array(msg.position[:9], dtype=np.float32)
        if len(q) == 9:
            self.latest_q = q

    def tick(self):
        # 图像和关节都到了才能推理
        if self.latest_img is None or self.latest_q is None:
            return

        # 每 exec_k 步重新推理一次
        if self.chunk_actions is None or self.chunk_idx >= self.exec_k:
            self.img_buf.append(self.latest_img)
            self.q_buf.append(self.latest_q)
            while len(self.img_buf) < self.history:
                self.img_buf.appendleft(self.latest_img)
                self.q_buf.appendleft(self.latest_q)

            self.chunk_actions = self.infer()
            self.chunk_idx = 0
            self.infer_count += 1

        # 取当前这一步动作
        action = self.chunk_actions[self.chunk_idx]
        self.chunk_idx += 1

        # 发布
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"c{i}" for i in range(8)]
        msg.position = [float(v) for v in action]
        self.pub.publish(msg)
        self.pub_count += 1

        if self.pub_count % 40 == 0:
            self.get_logger().info(
                f"pub={self.pub_count} infer={self.infer_count} "
                f"grip={action[7]:.0f} j1={action[0]:+.3f}"
            )

    def infer(self):
        imgs = np.stack(list(self.img_buf)).astype(np.float32) / 255.0
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).unsqueeze(0)

        qs = np.stack(list(self.q_buf)).astype(np.float32)
        qs = (qs - self.q_mean) / self.q_std
        qs = torch.from_numpy(qs.reshape(1, -1).astype(np.float32))

        with torch.no_grad():
            pred = self.model(imgs, qs)[0].numpy()

        return pred * self.a_std + self.a_mean


def main():
    args = parse_args()
    rclpy.init()
    node = PolicyNode(args)
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
