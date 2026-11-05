# MuJoCo 机械臂抓取 · 模仿学习（ACT）

在 MuJoCo 中搭建 Franka 抓取仿真，用脚本化专家采集演示数据，训练 ACT（Action Chunking Transformer）策略，并在随机方块位置下做闭环评估。

## 结果

| 指标 | 数值 |
|---|---|
| 闭环成功率（50 次随机位置） | **84%（42/50）** |
| 严格口径（抬升 > 0.5 m） | 52%（26/50） |
| 基线：单帧 MLP 行为克隆 | 30%（30 次） |
| 专家（脚本）成功率 | 100% |

成功定义：一次 episode 结束时，方块仍被夹持在 `final_z > 0.47 m`（初始中心 0.43 m，即抬离桌面 > 4 cm）。严格口径额外要求过程中最高点超过 0.5 m。

## 方法

```
MuJoCo 场景 (5cm 方块, 随机位置)
        │
        ▼
脚本化专家：IK（阻尼最小二乘） + 路标 + 夹爪时序
        │  100 条 episode, 20Hz
        ▼
数据集：128×128 图像 + 9 维 qpos + 8 维动作
        │
        ▼
ACT-lite：4 帧图像/qpos 历史 → CNN → Transformer
          → 16 个 query → 16 步动作
        │
        ▼
闭环评估：每 8 步重规划，50 次随机位置
```

关键设计：

- **IK**：`Jᵀ(JJᵀ + λI)⁻¹ e`，阻尼最小二乘，λ=1e-3
- **动作空间**：7 个关节目标角 + 1 个夹爪指令（0–255）
- **ACT-lite**：简化版 ACT，去掉 VAE，L1 loss，chunk=16，history=4
- **评估判据**：`final_z > 0.47`（抬离并保持），同时报告严格口径

## 复现

环境：Ubuntu 20.04 + RTX 3080 12GB + PyTorch 2.5.1 + MuJoCo 3.17

```bash
# 1. 采集专家数据（本地 WSL，约 15 分钟）
python 08_record.py

# 2. 训练 ACT（服务器，约 20 分钟）
python 11_train_act.py --data ~/robot-project/data/raw --epochs 60 --batch 64 --device cuda

# 3. 闭环评估（服务器）
python 12_eval_act.py --episodes 50 --exec-k 8 --device cuda
```

## 目录

```
robot-project/
├── 04_pick_place.py        # 早期：固定路标抓取
├── 05_jacobian_check.py    # 雅可比数值验证
├── 07_collect.py           # 抓取流程测试
├── 08_record.py            # 专家数据采集
├── act_model.py            # ACT-lite 模型
├── 11_train_act.py         # 训练
├── 12_eval_act.py          # 闭环评估
├── bc_model.py             # 基线：MLP 行为克隆
├── 09_train_bc.py          # 基线训练
├── 10_eval.py              # 基线评估
├── docs/
│   ├── glossary.md         # 知识树笔记
│   ├── pipeline.md         # 全流程框架
│   └── jobs.md             # 岗位地图
└── data/raw/               # 100 条专家数据
```

## 失败分析

- 主要失败集中在工作空间边缘（x < 0.48 或 x > 0.64）：机械臂接近伸展，关节误差放大
- 部分失败是抬升高度差 1–7 mm（严格口径下判失败，实际已抓离桌面）
- `exec_k=16`（一次执行 0.8 s）显著变差，说明闭环重规划频率很重要

## 局限与后续

- 仅仿真，未做 sim2real
- 未使用时序集成（temporal ensemble）
- 时延鲁棒性实验（通信背景的差异化方向）尚未开展

## 参考

- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT), RSS 2023
- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie
