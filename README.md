# MuJoCo 抓取系统：ACT 模仿学习 + ROS2 多节点架构

在 MuJoCo 中搭建 Franka 抓取仿真，用脚本化专家采集演示数据，训练 ACT 策略，并改造为 ROS2 多节点系统，用于研究**控制链路延迟**与**泛化性**对抓取性能的影响。

## 结果

| 实验 | 指标 | 结果 |
|---|---|---|
| 静态抓取（单脚本） | 闭环成功率 | 84%（50 次） |
| 静态抓取（ROS2 闭环） | 闭环成功率 | 84% |
| 时延影响（0–1000ms） | 成功率 | 100%（无显著影响） |
| 尺寸泛化（训练 5cm） | 4cm / 5cm / 6cm / 7cm | 60% / 80% / 0% / 0% |

**关键发现**：

- **恒定延迟对准静态抓取几乎无影响**：1000ms + 20% 丢包仍 100% 成功。原因是策略输出目标姿势序列，延迟只让序列整体平移，终态不变。
- **尺寸零泛化**：训练分布（5cm）之外，6cm 即完全失效（0%）。策略记忆的是针对特定尺寸的动作序列，而非通用抓取技能。

详细数据见 [`docs/experiments.md`](docs/experiments.md)。

## 系统架构

```
┌──────────────┐  /joint_states   ┌──────────────┐
│  sim_node    │  /camera/image   │ policy_node  │
│  (Python)    │ ────────────────▶│ ACT 推理      │
│  MuJoCo 20Hz │                  │ (PyTorch)    │
└──────┬───────┘                  └──────┬───────┘
       │                                 │ /joint_command
       │ /joint_command_delayed          │
       │                                 ▼
       │                        ┌──────────────────┐
       └────────────────────────│  latency_node    │
                                │  (C++/rclcpp)    │
                                │  延迟/抖动/丢包   │
                                └──────────────────┘
```

### 节点职责

| 节点 | 语言 | 职责 |
|---|---|---|
| `sim_node` | Python | MuJoCo 物理推进、图像渲染、发布状态 |
| `policy_node` | Python | ACT 推理，20Hz，exec_k=8 |
| `latency_node` | **C++** | 时延注入（延迟/抖动/丢包）、端到端延迟统计 |
| `command_node` | **C++** | 固定目标角发布（测试用） |
| `teleop_node` | Python | 订阅状态并打印（测试用） |

## 技术栈

- **仿真**：MuJoCo 3.17，MJCF，EGL 离屏渲染
- **框架**：ROS2 Humble，rclcpp（C++），rclpy（Python）
- **学习**：PyTorch 2.5.1，ACT-lite（CNN + Transformer encoder/decoder）
- **算法**：阻尼最小二乘 IK、行为克隆、动作分块
- **工程**：Linux（WSL2）、colcon/CMake、Git、tmux

## 复现

### 1. 训练（需要 GPU）

```bash
python 08_record.py                                    # 采集 100 条演示
python 11_train_act.py --data data/raw --epochs 60     # 训练 ACT
python 12_eval_act.py --episodes 50 --exec-k 8         # 闭环评估
```

### 2. ROS2 系统（CPU 即可）

```bash
# 终端 A：仿真
python3 ros2/sim_node_camera.py --cmd-topic /joint_command_delayed

# 终端 B：策略
python3 ros2/policy_node.py

# 终端 C：时延节点
cd ros2_ws && colcon build --packages-select robot_cpp
source install/setup.bash
ros2 run robot_cpp latency_node --ros-args -p delay_ms:=200
```

### 3. 批量实验

```bash
python3 run_experiment.py 5    # 每组 5 次，自动出汇总表
```

## 目录

```
robot-project/
├── 04_pick_place.py          # 早期：固定路标抓取
├── 05_jacobian_check.py      # 雅可比数值验证
├── 08_record.py              # 专家数据采集
├── 09_train_bc.py            # 基线：MLP 行为克隆
├── 10_eval.py                # 基线评估
├── 11_train_act.py           # ACT 训练
├── 12_eval_act.py            # ACT 评估
├── act_model.py              # ACT-lite 模型
├── run_experiment.py         # 批量实验脚本
├── ros2/                     # Python 节点
│   ├── sim_node_camera.py
│   ├── policy_node.py
│   ├── command_node.py
│   └── teleop_node.py
├── ros2_ws/src/robot_cpp/    # C++ 节点
│   ├── src/command_node.cpp
│   └── src/latency_node.cpp
├── docs/
│   ├── experiments.md        # 实验记录（含失败案例）
│   ├── pipeline.md           # 全流程框架
│   ├── glossary.md           # 知识树笔记
│   └── retrospective.md      # 项目复盘
└── data/raw/                 # 100 条专家数据
```

## 关键设计

**IK（专家）**

```python
dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e)
```

阻尼最小二乘，λ=1e-3，防止奇异位形处关节速度爆炸。

**ACT-lite**

```
4 帧图像 + 4 帧 qpos
  → CNN（保留 4×4 空间特征）
  → Transformer Encoder（时序建模）
  → 16 个可学习 query → Transformer Decoder
  → 16 步动作
```

**评估判据**

- 成功：结束时方块仍被夹持在 `初始高度 + 4cm` 以上
- 同时报告 `max_z`（过程最高点）和 `min_z_after_grasp`（抓住后最低点），识别"抬起来又掉"

## 局限

- 仅仿真，未做 sim2real
- 尺寸泛化零容忍（6cm 即失效）
- 恒定延迟对准静态任务无影响（需要动态任务才能测出延迟敏感性）
- 未使用时序集成（temporal ensemble）

## 参考

- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT), RSS 2023
- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie
- ROS2 Humble: https://docs.ros.org/en/humble/
