// command_node.cpp — C++ 版发令节点
//
// 功能：以 10 Hz 往 /joint_command 发 8 个控制量（7 个手臂关节 + 1 个夹爪）。
// 和 Python 版 ros2/command_node.py 做的事情完全一样，只是用 C++ 写。
//
// 编译运行：
//   cd ~/robot-project/ros2_ws
//   colcon build --packages-select robot_cpp
//   source install/setup.bash
//   ros2 run robot_cpp command_node

#include <chrono>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

using namespace std::chrono_literals;

// 目标姿势：7 个手臂关节角 + 1 个夹爪（255 = 张开，0 = 闭合）
static const std::vector<double> TARGET = {
    0.0, 0.0, 0.0, -1.0, 0.0, 1.5, -0.78, 255.0};

class CommandNode : public rclcpp::Node {
 public:
  CommandNode() : Node("command_node_cpp") {
    // 创建发布者：话题 /joint_command，队列 10
    pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
        "/joint_command", 10);

    // 10 Hz 定时器，每 100ms 调用一次 tick()
    timer_ = this->create_wall_timer(
        100ms, std::bind(&CommandNode::tick, this));

    RCLCPP_INFO(this->get_logger(),
                "command_node (C++) started, publishing %zu values at 10 Hz",
                TARGET.size());
  }

 private:
  void tick() {
    auto msg = sensor_msgs::msg::JointState();
    msg.header.stamp = this->now();

    // 填充关节名：c0, c1, ..., c7
    for (size_t i = 0; i < TARGET.size(); ++i) {
      msg.name.push_back("c" + std::to_string(i));
    }
    msg.position = TARGET;  // std::vector<double> 直接赋值

    pub_->publish(msg);

    count_++;
    if (count_ % 20 == 0) {
      RCLCPP_INFO(this->get_logger(), "sent %d, joint4=%.2f", count_,
                  TARGET[3]);
    }
  }

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  int count_ = 0;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);              // 初始化 ROS2
  auto node = std::make_shared<CommandNode>();
  rclcpp::spin(node);                    // 一直转，处理定时器
  rclcpp::shutdown();                    // 退出时清理
  return 0;
}
