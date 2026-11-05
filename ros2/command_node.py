"""command_node.py — 发目标关节角到 /joint_command。

职责：以 10 Hz 发送 8 个控制量（7 个手臂关节 + 1 个夹爪）。
先发一个和 home 不同的姿势，用来验证 sim_node 能不能动起来。

运行：
    source /opt/ros/humble/setup.bash
    python3 ros2/command_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

CTRL_HZ = 10.0

# 目标姿势：7 个手臂关节角 + 1 个夹爪（255 = 张开，0 = 闭合）
# 和 home 的区别：第 4 个关节从 -1.57 改成 -1.0
TARGET = [0.0, 0.0, 0.0, -1.0, 0.0, 1.5, -0.78, 255.0]


class CommandNode(Node):
    def __init__(self):
        super().__init__("command_node")

        self.pub = self.create_publisher(JointState, "/joint_command", 10)
        self.timer = self.create_timer(1.0 / CTRL_HZ, self.tick)

        self.count = 0
        self.get_logger().info(
            f"command_node started, publishing {len(TARGET)} values at {CTRL_HZ} Hz"
        )

    def tick(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"c{i}" for i in range(len(TARGET))]
        msg.position = [float(v) for v in TARGET]

        self.pub.publish(msg)
        self.count += 1

        if self.count % 20 == 0:
            self.get_logger().info(f"sent {self.count} commands, joint4={TARGET[3]:.2f}")


def main():
    rclpy.init()
    node = CommandNode()
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
