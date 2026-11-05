# 工程能力补强计划：Git / ROS2 / C++

> 目标：在 2027 年秋招前补齐机器人工程岗的硬门槛，并且用**同一个抓取项目**承载，不另起炉灶。
> 时间：14 周，每周 10–15 小时。

## 为什么是这三样

| 技能 | 重要度 | 原因 |
|---|---|---|
| C++ | 高 | 机器人工程岗笔试、面试、实际代码都考 |
| ROS2 | 高 | 机器人软件岗的通用框架，不会很难投 |
| Git 协作 | 中 | 团队协作基础，不难，但没有会显得不专业 |

顺序：**Git（1 周）→ ROS2（3 周）→ C++（6 周）→ C++ ROS2 节点（3 周）→ 开源 PR（1 周）**。

---

## 第 1 周：Git 与 GitHub

### 学什么

- `init / add / commit / status / log / diff`
- 分支：`branch / checkout / merge`
- 远程：`remote / push / pull`
- `.gitignore`、README
- 什么是 PR（Pull Request）

### 做什么

把现有 `robot-project` 推上 GitHub：

```bash
cd ~/robot-project
git init
# .gitignore 已包含 data/、*.npz、*.mp4、.vscode/
git add .
git commit -m "init: MuJoCo pick-and-place with ACT"
# 在 GitHub 建仓库后
git remote add origin git@github.com:<你的用户名>/robot-project.git
git push -u origin main
```

### 验收

- GitHub 上有仓库，至少 10 次 commit
- README 能让人看懂项目、结果、复现步骤
- 数据和模型没有进 git

---

## 第 2–4 周：ROS2（Python）

### 环境

- 本地 WSL Ubuntu 22.04 → 装 **ROS2 Humble**
- 服务器 Ubuntu 20.04 → 对应 Foxy，已 EOL；学习阶段先在 WSL 做，部署以后再说

### 学什么

- 节点（node）、话题（topic）、服务（service）、参数（parameter）
- `rclpy` 写发布者和订阅者
- launch 文件
- tf2：坐标变换
- `ros2 topic echo / pub / node list`

### 做什么：把抓取项目拆成三个节点

```
/sim_node        发布 /joint_states 和 /camera/image
/policy_node     订阅上面两个，发布 /joint_command
/robot_node      订阅 /joint_command，驱动物理仿真
```

先不接真实 MuJoCo，用假数据跑通通信；再把 MuJoCo 包进 `sim_node`。

### 验收

- `ros2 launch` 一次启动三个节点
- `ros2 topic echo /joint_command` 能看到策略输出
- 能用 `ros2 run` 单独跑每个节点

---

## 第 5–10 周：C++

### 学什么

- 基础：类型、引用、指针、const
- 面向对象：类、构造/析构、继承
- STL：vector、map、string、algorithm
- 内存：栈/堆、RAII、智能指针
- CMake：`CMakeLists.txt`、编译、链接
- 调试：gdb 基础、打印、断言

### 做什么

1. 把一个小 Python 脚本改写成 C++（比如数据统计、简单控制器）
2. 写一个 CMake 工程，能编译、能运行
3. 再写一个 C++ 的 ROS2 节点，替代 Python 的 `policy_node` 里的通信部分

### 验收

- 能独立写 `CMakeLists.txt`
- C++ 节点能发布/订阅
- 能解释指针和引用的区别、RAII 是什么

---

## 第 11–13 周：C++ + ONNX 推理节点

### 做什么

1. 把 ACT 模型导出为 ONNX
2. C++ 节点用 ONNX Runtime 加载模型
3. 订阅图像和关节角，推理，发布动作
4. 和 Python 版对比输出，确认一致

### 验收

- C++ 节点在仿真里跑通抓取
- 推理频率 ≥ 10 Hz
- 能解释 ONNX 的作用

---

## 第 14 周：开源 PR

### 做什么

找一个 `good first issue`：

- LeRobot（HuggingFace）
- ROS2 文档或示例
- MuJoCo Menagerie

改一个小 bug、补一段文档，提 PR。

### 验收

- PR 被合并，或至少收到 review
- 简历上可以写“向 XX 提交 PR”

---

## 时间线与检查点

| 周 | 内容 | 检查点 |
|---|---|---|
| 1 | Git + GitHub | 仓库上线 |
| 2–4 | ROS2 Python | 三节点跑通 |
| 5–10 | C++ 基础 | CMake 工程能跑 |
| 11–13 | C++ + ONNX | C++ 节点抓取成功 |
| 14 | 开源 PR | PR 提交 |

## 资源

- Git：Pro Git（前 3 章）
- ROS2：官方 Tutorials（Humble）
- C++：learncpp.com（系统）+ cppreference（查）
- CMake：官方 tutorial
- ONNX Runtime：官方 C++ 示例

## 注意

- 不要为了学 C++ 另起一个全新项目，**在现有抓取项目上扩展**，故事才连贯。
- 每完成一个阶段，更新 README 和简历，把“学习中”改成“掌握”。
- 简历上写 C++/ROS2 之前，先确保能接住三层追问。
