# 新项目计划：ROS2 遥操作抓取系统与时延鲁棒性基准

> 目标：在现有 MuJoCo + ACT 抓取项目上，扩展出一个 ROS2 系统，同时学会 ROS2、C++、Git 协作，
> 并做出一个通信背景的差异化成果：**时延对抓取成功率的影响与补偿**。

## 1. 项目一句话

把“单脚本抓取”改造成“ROS2 多节点系统”，在遥操作/策略控制链路中注入时延、抖动、丢包，测量抓取成功率的变化，并比较不同 QoS 配置。

## 2. 为什么这个项目适合你

- 复用现有 MuJoCo 场景、ACT 模型、评估流程，不用从零开始
- ROS2 是机器人软件岗的硬门槛
- C++ 节点可以从简单的时延注入器开始，逐步过渡到 ONNX 推理
- 时延/通信是你的本行，能做出别人没有的差异化结果

## 3. 系统架构

```
                 ┌──────────────┐
                 │ teleop_node  │  键盘/脚本路标
                 └──────┬───────┘
                        │ /cmd_joint
                        ▼
                 ┌──────────────┐
                 │ latency_node │  注入时延/抖动/丢包，统计端到端延迟
                 └──────┬───────┘
                        │ /cmd_joint_delayed
                        ▼
                 ┌──────────────┐        /joint_states
                 │   sim_node   │ ──────────────────────┐
                 │   (MuJoCo)   │        /camera/image  │
                 └──────┬───────┘        /cube_pose    │
                        │                              ▼
                        │                       ┌──────────────┐
                        └──────────────────────▶│ policy_node  │
                                                │ Python→C++   │
                                                └──────┬───────┘
                                                       │ /cmd_joint
                                                       ▼
                                                （回到 latency_node）
```

### 节点职责

| 节点 | 语言 | 职责 |
|---|---|---|
| `sim_node` | Python | 加载 MuJoCo，推进物理，发布关节/图像/方块位姿，订阅命令 |
| `teleop_node` | Python | 键盘或脚本路标，发布目标关节角 |
| `latency_node` | **C++** | 延迟队列、抖动、丢包、时间戳统计 |
| `policy_node` | Python → **C++** | ACT 推理，输出 16 步动作 |
| `eval_node` | Python | 跑 episode、统计成功率、写 CSV |
| `tf2` | — | world / base / hand / cube 坐标变换 |

### 话题与消息

| 话题 | 类型 | 方向 |
|---|---|---|
| `/joint_command` | `sensor_msgs/JointState` 或自定义 8 维 | 控制 |
| `/joint_states` | `sensor_msgs/JointState` | 状态 |
| `/camera/image` | `sensor_msgs/Image` | 感知 |
| `/cube_pose` | `geometry_msgs/PoseStamped` | 真值（评估用） |
| `/latency_stats` | 自定义 msg | 统计 |

## 4. 里程碑

### M1（第 1 周）：ROS2 骨架 + Git

- 安装 ROS2 Humble（WSL Ubuntu 22.04）
- 建仓库，`main` + 分支保护
- 写两个最小节点：publisher / subscriber，`ros2 run` 跑通
- 写 launch 文件

**验收**：`ros2 launch` 启动两个节点，`ros2 topic echo` 能看到消息；仓库有 5 次以上 commit。

### M2（第 2–3 周）：sim_node + teleop_node

- 把 MuJoCo 包进 `sim_node`，10–20 Hz 发布 `/joint_states` 和 `/camera/image`
- `teleop_node` 发布固定路标序列
- 端到端跑通一次抓取

**验收**：`ros2 topic hz` 显示稳定频率；抓取成功。

### M3（第 4 周）：tf2 + QoS + 记录

- 发布 world → base → hand → cube 的 tf
- 配置 QoS：reliable vs best-effort，depth 1/10
- 用 `ros2 bag record` 录一段，回放验证

**验收**：`ros2 run tf2_ros tf2_echo` 能查到 hand 相对 cube 的位置。

### M4（第 5–6 周）：C++ latency_node

- 学 rclcpp、CMake、参数
- 用 C++ 重写时延注入节点：固定延迟、随机抖动、丢包率
- 发布 `/latency_stats`

**验收**：C++ 节点可配置延迟 0/50/100/200 ms；端到端延迟测量误差 < 5 ms。

### M5（第 7–8 周）：C++ ONNX policy_node

- 把 ACT 导出 ONNX（固定 history=4、chunk=16）
- C++ 节点用 ONNX Runtime 加载，订阅图像/状态，发布动作
- 与 Python 版输出对比

**验收**：C++ 节点在仿真里抓取成功；推理频率 ≥ 10 Hz；与 Python 输出误差 < 1e-3。

### M6（第 9–10 周）：时延实验

实验矩阵：

| 变量 | 取值 |
|---|---|
| 固定延迟 | 0 / 50 / 100 / 200 ms |
| 抖动 | 0 / ±20 ms |
| 丢包 | 0 / 5% / 10% |
| QoS | reliable / best-effort |

记录：成功率、平均完成时间、失败模式。

**验收**：得到“延迟–成功率”曲线；至少 30 次/组。

### M7（第 11–12 周）：收尾

- README：架构图、实验表、复现步骤
- demo 视频：正常 vs 200 ms 延迟
- 复盘文档
- 简历条目
- （可选）给 LeRobot 或 ROS2 提一个小 PR

## 5. 你在每个阶段学到什么

| 阶段 | ROS2 | C++ | Git | 通信 |
|---|---|---|---|---|
| M1 | 节点、话题、launch | — | init/commit/branch | — |
| M2 | 发布/订阅、定时器 | — | 分支开发 | — |
| M3 | tf2、QoS、rosbag | — | PR、review | QoS 语义 |
| M4 | 参数、自定义 msg | rclcpp、CMake、智能指针 | 分支合并 | 延迟/抖动建模 |
| M5 | 组件化 | ONNX Runtime、内存管理 | 版本管理 | 推理延迟 |
| M6 | 多节点联调 | 性能分析 | tag/release | 时延-成功率 |
| M7 | 文档 | — | PR | 实验设计 |

## 6. 交付物

- GitHub 仓库：代码、README、架构图、实验结果
- demo 视频：正常链路 vs 高延迟链路
- 实验报告：延迟/抖动/丢包/QoS 对成功率的影响
- 简历条目：

> 将 MuJoCo 抓取系统改造为 ROS2 多节点架构（仿真、遥操作、策略、时延注入），用 C++/rclcpp 实现时延注入与统计节点，基于 ONNX Runtime 完成 ACT 的 C++ 推理部署；系统测量 0–200 ms 时延下抓取成功率，比较 reliable/best-effort QoS，定位链路瓶颈。

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| WSL 里 ROS2 图形/显示问题 | 全程 headless，图像走话题，不依赖窗口 |
| ONNX 导出失败 | 先用固定 dummy input 导出，逐层对比 |
| C++ 学习曲线 | 先写时延节点（逻辑简单），再写推理节点 |
| 时间不够 | 先完成 M1–M3（Python 版 ROS2），C++ 可以后补 |
| 实验太多 | 先做固定延迟四组，再加抖动/丢包 |

## 8. 第一周就做这三件事

1. 在 WSL 装 ROS2 Humble，跑通官方 talker/listener
2. 建 GitHub 仓库，把现有 `robot-project` 推上去
3. 写第一个 `sim_node` 的骨架：定时器 10 Hz 发布假 `/joint_states`

做完发我仓库地址和 `ros2 topic hz` 的截图。
