# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

这是 Global Humanoid Robot Challenge 2026 的官方基线，基于 LeRobot 0.5.1，面向 WalkerS2 人形机器人仿真。完整流程是：下载 Isaac Sim 资产和数据集，启动 Isaac Sim 5.1 Docker 环境，在仿真中遥操作或自动采集 LeRobotDataset V3.0 数据，训练 ACT / Diffusion / Pi0 / Pi0.5 / SmolVLA 等策略，再进行 replay 或策略评估。

仓库有两层主要代码：

- `src/lerobot/`：LeRobot 主包。包含数据集 V3、相机/机器人/遥操作器抽象、训练与评估脚本、策略实现以及本项目新增的 WalkerS2 仿真实现。
- `Ubtech_sim/`：较早的 Isaac Sim 场景入口和底层场景模块。`main.py` 编排仿真，`source/SceneBuilder.py` 构建场景，`RobotArticulation.py` 控制机器人，`DualArmIK.py` 做双臂逆运动学，`grasp_planner.py` 做抓取规划，`DataLogger.py` 输出评分 CSV 和相机 HDF5 数据，`config/*.yaml` 保存任务配置。

## 环境与依赖

- 推荐通过 `Dockerfile` 构建在 NVIDIA Isaac Sim 5.1.0 基础镜像上运行。宿主机需要 NVIDIA 驱动、CUDA 12.8、Docker、NVIDIA Container Toolkit；README 建议使用 580 系列驱动。
- 仿真依赖 Isaac Sim、Pinocchio、PyYAML，并要求 `assets/` 中的比赛资产。训练和运行还需要 GPU、PyTorch、Transformers 等依赖。
- `pyproject.toml` 是 Python 包和依赖的权威配置，使用 `src` layout；可选依赖包括 `pi`、`smolvla`、`dev`、`test` 等。Dockerfile 使用 `/isaac-sim/python.sh`，并刻意固定 Isaac Sim 兼容的 numpy、packaging、torchcodec 等版本，不要随意改成系统 Python 的安装方式。
- 大型资源不在普通源码中：仿真资产来自 Hugging Face `UBTECH-Robotics/challenge2026_assets`，数据集来自 `UBTECH-Robotics/challenge2026_dataset`。修改仿真路径或配置时确认本地资产确实已下载。

## 常用命令

以下命令默认在仓库根目录执行；进入容器后优先使用 `/isaac-sim/python.sh`。

### 安装与构建

```bash
# 普通 Python 开发环境（不提供 Isaac Sim）
python -m pip install -e '.[dev,test]'

# 构建 Isaac Sim 镜像
docker build -t ghrc2026-baseline .

# 运行容器时需按宿主机情况挂载 X11、GPU、设备和数据目录；Dockerfile 默认工作目录为 /workspace/GlobalHumanoidRobotChallenge_2026_Baseline
```

Dockerfile 已包含 Isaac Sim Python 环境的依赖安装和 editable install。涉及仿真、evdev、相机或 GPU 时，不要用普通 `python` 代替容器中的 `/isaac-sim/python.sh`。

### 质量检查与测试

仓库当前没有 `tests/` 目录、pytest 配置或 CI 测试工作流。若环境已安装开发依赖，可运行：

```bash
ruff check .
ruff format --check .
pytest

# 运行单个测试文件、测试函数（若后续新增测试）
pytest path/to/test_file.py
pytest path/to/test_file.py::test_name
```

对仅依赖纯 Python 的改动，至少运行相关 `ruff check path/to/file.py`；对 Isaac Sim 相关改动，应在目标容器中做导入或启动验证，因为本地普通 Python 通常无法导入 `isaacsim`。

### 仿真、遥操作与数据流程

```bash
# 直接启动旧版 Ubtech_sim 入口（需要 Isaac Sim 环境和 assets）
python Ubtech_sim/main.py

# LeRobot WalkerS2 键盘遥操作
lerobot-teleoperate \
  --robot.type=walker_s2_sim \
  --robot.headless=false \
  --teleop.type=walker_s2_keyboard \
  --task=packing_box \
  --display_data=true

# 记录数据（参数按任务和数据集调整）
lerobot-record --robot.type=walker_s2_sim --teleop.type=walker_s2_keyboard

# 回放数据集
lerobot-replay --robot.type=walker_s2_sim --dataset.repo_id=<user>/<dataset>

# 自动数采；可选择 Part_Sorting、Foam_Inlaying、Packing_Box、Conveyor_Sorting 等任务
/isaac-sim/python.sh -m lerobot.scripts.auto_collect_main \
  --robot.type=walker_s2_sim \
  --auto_collect.task=Part_Sorting \
  --auto_collect.repo_id=<user>/<dataset> \
  --auto_collect.num_episodes=10 \
  --auto_collect.record_data=true

# 分段传送带自动数采入口
/isaac-sim/python.sh -m lerobot.scripts.auto_collect_conveyor_segmented \
  --robot.type=walker_s2_sim
```

`lerobot-teleoperate`、`lerobot-record`、`lerobot-replay` 会通过注册的 `RobotConfig` / `TeleoperatorConfig` 创建对象。WalkerS2 的注册和实现位于 `src/lerobot/robots/walker_s2_sim/`，键盘遥操作位于 `src/lerobot/teleoperators/walker_s2_keyboard/`；配置通过 `--robot.*`、`--teleop.*`、`--task` 等 draccus 参数传入。自动数采由 `src/lerobot/auto_collect/` 的任务类和 `src/lerobot/scripts/auto_collect_*.py` 入口组成。

### 训练与评估

```bash
lerobot-train --help
lerobot-eval --help

# 训练示例：具体 policy、dataset、batch size 和输出目录按实验配置
lerobot-train \
  --dataset.repo_id=<user>/<dataset> \
  --policy.type=pi05

# 查看数据集或本地数据
lerobot-info --help
lerobot-dataset-viz --help
```

Pi0 / Pi0.5 等策略通常需要 `pip install -e '.[pi]'` 或对应可选依赖。训练时还可能依赖 Hugging Face 登录和 Weights & Biases 配置；不要把凭据写入仓库。

## 代码架构与修改位置

- **机器人生命周期**：脚本入口解析 draccus 配置，调用 `src/lerobot/robots/utils.py:make_robot_from_config` 创建机器人；WalkerS2 实现负责启动/关闭 Isaac Sim、建立世界和物理/渲染回调、读取观测并接收动作。
- **WalkerS2 仿真适配**：`src/lerobot/robots/walker_s2_sim/` 是 LeRobot 与 Isaac Sim 之间的适配层，包含配置、Articulation/场景接口、相机可视化和动作执行。涉及仿真控制时优先修改这一层，而不是直接在脚本入口中塞控制逻辑。
- **遥操作**：`src/lerobot/teleoperators/walker_s2_keyboard/` 将键盘事件转换为控制输入；WalkerS2 的控制是回调驱动的，record/teleoperate 脚本会将 teleop 连接到机器人回调。
- **数据集与训练**：`src/lerobot/datasets/` 管理 LeRobotDataset V3 格式，`src/lerobot/scripts/lerobot_record.py` 负责按 episode 采集，`lerobot_train.py` 负责 dataloader、策略更新、checkpoint 和可选评估，`src/lerobot/policies/` 实现各策略。
- **自动数采**：`src/lerobot/auto_collect/auto_collect_base.py` 提供通用采集状态机和输出逻辑，各任务文件实现任务特定的场景初始化、轨迹规划和 episode 流程。`Ubtech_sim/config/*.yaml` 与 `src/lerobot/auto_collect/` 的任务配置需要保持语义一致。
- **旧版仿真入口**：`Ubtech_sim/main.py` 是独立的 Isaac Sim 应用入口；其 `source/` 模块负责场景、坐标变换、IK、抓取和日志。只有需要使用这套旧入口或评分数据记录时才直接修改它。
- **配置和注册机制**：机器人、遥操作器和策略通过配置基类的 `register_subclass` 注册。新增实现时要同时提供配置类、实现类和包导入注册，确保 `make_*_from_config` 能发现类型；不要只新增一个未注册的脚本。

## 重要约定

- 任务配置中的资产路径通常是相对于 `assets/resources/` 的路径；修改 `root_path`、USD 或 URDF 路径时同时检查 Docker 工作目录和宿主机挂载路径。
- Isaac Sim 的核心依赖对 numpy、packaging、torchcodec、USD 版本敏感。优先遵循 Dockerfile 的版本约束，避免在容器内执行会升级这些基础包的无约束安装。
- 录制、回放和推理可能会创建 `outputs/`、数据集目录、CSV/HDF5 和视频等大型文件；这些通常不应提交到 Git。
- README 的完整流程文档位于 `https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/`，包括资源下载、环境设置、快速开始、训练、推理、四相机显示和自动数采七个阶段。
