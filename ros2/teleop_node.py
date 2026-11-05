"""teleop_node.py — 订阅 /joint_states，打印收到的关节角度。

这是「收音机」：一直开着，有消息来了就打印。

运行：
    source /opt/ros/humble/setup.bash
    python3 ros2/teleop_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class TeleopNode(Node):
    def __init__(self):
        super().__init__("teleop_node")

        # 订阅者：从 /joint_states 收 JointState，收到就调用 on_joint
        self.sub = self.create_subscription(
            JointState,       # 消息格式
            "/joint_states",  # 话题名
            self.on_joint,    # 回调函数（注意：不写括号）
            10,               # 队列长度
        )

        self.count = 0
        self.get_logger().info("teleop_node started, waiting for /joint_states")

    def on_joint(self, msg):
        """每收到一条消息，ROS2 自动调用这个函数一次。"""
        self.count += 1

        # 每 10 条打印一次，避免刷屏
        if self.count % 10 == 0:
            names = ", ".join(msg.name[:3])
            vals = ", ".join(f"{p:+.3f}" for p in msg.position[:3])
            self.get_logger().info(f"#{self.count} {names}: {vals}")


def main():
    rclpy.init()
    node = TeleopNode()
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
