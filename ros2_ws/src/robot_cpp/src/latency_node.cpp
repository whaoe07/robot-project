// latency_node.cpp — 时延注入节点
//
// 功能：
//   订阅 /joint_command         （收原始命令）
//   延迟 delay_ms 毫秒后再发到 /joint_command_delayed
//   同时统计端到端延迟、队列长度
//
// 为什么不用 sleep：
//   一个节点要同时处理多条消息，sleep 会卡住整个线程。
//   这里用「队列 + 定时器」：收到时记下到期时间，定时器只发到期的。
//
// 运行（200ms 延迟）：
//   ros2 run robot_cpp latency_node --ros-args -p delay_ms:=200
//
// 参数：
//   delay_ms  固定延迟（毫秒），默认 0
//   jitter_ms 随机抖动幅度（±毫秒），默认 0
//   drop_rate 丢包率 0~1，默认 0

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <memory>
#include <random>
#include <string>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

using namespace std::chrono_literals;
using JointState = sensor_msgs::msg::JointState;

// 队列里的一项：什么时候该发 + 发什么
struct Pending {
  rclcpp::Time due;      // 到期时间
  rclcpp::Time arrived;  // 到达时间（用于统计延迟）
  JointState msg;
};

class LatencyNode : public rclcpp::Node {
 public:
  LatencyNode() : Node("latency_node") {
    // 声明参数，运行时可改
    delay_ms_ = this->declare_parameter<int>("delay_ms", 0);
    jitter_ms_ = this->declare_parameter<int>("jitter_ms", 0);
    drop_rate_ = this->declare_parameter<double>("drop_rate", 0.0);

    in_topic_ = this->declare_parameter<std::string>("in_topic", "/joint_command");
    out_topic_ =
        this->declare_parameter<std::string>("out_topic", "/joint_command_delayed");

    pub_ = this->create_publisher<JointState>(out_topic_, 10);

    sub_ = this->create_subscription<JointState>(
        in_topic_, 10,
        [this](const JointState::SharedPtr msg) { on_command(msg); });

    // 每 10ms 检查一次队列有没有到期的
    timer_ = this->create_wall_timer(10ms, [this]() { flush(); });

    RCLCPP_INFO(this->get_logger(),
                "latency_node: %s -> %s, delay=%dms jitter=±%dms drop=%.2f",
                in_topic_.c_str(), out_topic_.c_str(), delay_ms_, jitter_ms_,
                drop_rate_);
  }

 private:
  void on_command(const JointState::SharedPtr msg) {
    received_++;

    // 丢包
    if (drop_rate_ > 0.0 && uniform_(rng_) < drop_rate_) {
      dropped_++;
      return;
    }

    // 实际延迟 = 固定延迟 + 随机抖动
    int total_ms = delay_ms_;
    if (jitter_ms_ > 0) {
      std::uniform_int_distribution<int> jitter(-jitter_ms_, jitter_ms_);
      total_ms += jitter(rng_);
    }
    if (total_ms < 0) total_ms = 0;

    auto now = this->now();
    Pending p;
    p.due = now + rclcpp::Duration::from_seconds(total_ms / 1000.0);
    p.arrived = now;
    p.msg = *msg;
    queue_.push_back(std::move(p));
  }

  void flush() {
    auto now = this->now();

    // 队列按到期时间有序：只看队头
    while (!queue_.empty() && queue_.front().due <= now) {
      auto& p = queue_.front();
      double latency_ms = (now - p.arrived).seconds() * 1000.0;
      pub_->publish(p.msg);
      sent_++;
      latency_sum_ms_ += latency_ms;
      queue_.pop_front();
    }

    // 每秒打印一次统计
    auto stamp = this->now();
    if ((stamp - last_report_).seconds() >= 1.0) {
      last_report_ = stamp;
      double avg = sent_ > 0 ? latency_sum_ms_ / sent_ : 0.0;
      RCLCPP_INFO(this->get_logger(),
                  "recv=%d sent=%d dropped=%d queue=%zu avg_latency=%.1fms",
                  received_, sent_, dropped_, queue_.size(), avg);
    }
  }

  // 参数
  int delay_ms_ = 0;
  int jitter_ms_ = 0;
  double drop_rate_ = 0.0;
  std::string in_topic_;
  std::string out_topic_;

  // ROS2 接口
  rclcpp::Publisher<JointState>::SharedPtr pub_;
  rclcpp::Subscription<JointState>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // 队列
  std::deque<Pending> queue_;

  // 统计
  int received_ = 0;
  int sent_ = 0;
  int dropped_ = 0;
  double latency_sum_ms_ = 0.0;
  rclcpp::Time last_report_ = this->now();

  // 随机数（抖动、丢包用）
  std::mt19937 rng_{std::random_device{}()};
  std::uniform_real_distribution<double> uniform_{0.0, 1.0};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LatencyNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
