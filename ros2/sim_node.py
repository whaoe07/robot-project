"""sim_node.py — 第一个 ROS2 节点骨架。

只做一件事：以 10 Hz 发布假的 /joint_states。
不接 MuJoCo，先验证节点、话题、定时器能跑通。

运行：
    source /opt/ros/humble/setup.bash
    python3 ros2/sim_node.py
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class SimNode(Node):
    def __init__(self):
        super().__init__("sim_node")

        # 发布者：话题名 /joint_states，队列长度 10
        self.pub = self.create_publisher(JointState, "/joint_states", 10)

        # 定时器：每 0.1 秒调用一次 tick()，即 10 Hz
        self.timer = self.create_timer(0.1, self.tick)

        self.t = 0.0
        self.get_logger().info("sim_node started, publishing at 10 Hz")

    def tick(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"joint{i}" for i in range(1, 8)] + ["finger1", "finger2"]

        # 用正弦波制造一点变化，方便在 topic echo 里看到数值在动
        msg.position = [math.sin(self.t + i) * 0.1 for i in range(7)] + [0.04, 0.04]
        msg.velocity = [0.0] * 9
        msg.effort = [0.0] * 9

        self.pub.publish(msg)
        self.t += 0.1

        if int(self.t * 10) % 10 == 0:
            self.get_logger().info(f"t={self.t:.1f}s published")


def main():
    rclpy.init()
    node = SimNode()
    try:
        rclpy.spin(node)  # 让节点一直活着，处理回调
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Ctrl+C 时 rclpy 已经自动 shutdown 过一次，再调用会报
        # "rcl_shutdown already called"。先判断再关。
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
