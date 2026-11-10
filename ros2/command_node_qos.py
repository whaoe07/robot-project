"""command_node_qos.py — 用 best-effort QoS 发命令，用于观察 QoS 不匹配。

QoS（Quality of Service，服务质量）决定消息怎么传：
  - reliable    : 保证送达，丢了重传（慢一点，但稳）
  - best-effort : 尽力送，丢了就丢（快，但不可靠）

用法：
    python3 ros2/command_node_qos.py                    # 默认 best-effort
    python3 ros2/command_node_qos.py --reliable         # 改成 reliable

实验：
  1. 用 best-effort 发，sim_node（默认 reliable 订阅）会收不到
  2. 加 --reliable 再发，就能收到
"""

import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

CTRL_HZ = 10.0
TARGET = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, -0.78, 255.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reliable", action="store_true", help="用 reliable 而不是 best-effort")
    return p.parse_args()


class CommandNodeQos(Node):
    def __init__(self, reliable=False):
        super().__init__("command_node_qos")

        policy = (
            ReliabilityPolicy.RELIABLE if reliable
            else ReliabilityPolicy.BEST_EFFORT
        )
        qos = QoSProfile(
            reliability=policy,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub = self.create_publisher(JointState, "/joint_command", qos)
        self.timer = self.create_timer(1.0 / CTRL_HZ, self.tick)
        self.count = 0

        self.get_logger().info(
            f"publishing with reliability={policy.name} at {CTRL_HZ} Hz"
        )

    def tick(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"c{i}" for i in range(len(TARGET))]
        msg.position = [float(v) for v in TARGET]
        self.pub.publish(msg)
        self.count += 1

        if self.count % 20 == 0:
            self.get_logger().info(f"sent {self.count}")


def main():
    args = parse_args()
    rclpy.init()
    node = CommandNodeQos(reliable=args.reliable)
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
