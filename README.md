# GoalFlow - End-to-End Autonomous Driving Framework

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

GoalFlow 是一个基于目标导向的端到端自动驾驶框架，采用 rectified flow  planning 技术和多模态感知融合来实现智能轨迹规划。

## 特性

- **多模态感知融合**: 基于 TransFuser 的 BEV (Bird's Eye View) 特征提取，融合相机和 LiDAR 数据
- **目标导向规划**: 通过目标词汇表 (Goal Vocabulary) 和目标评分器选择最优行驶目标
- **Rectified Flow Planning**: 使用 rectified flow 技术进行轨迹生成和优化
- **模块化设计**: 感知、目标选择、轨迹规划模块解耦，便于扩展和实验

## 项目结构

```
goalflow/
├── configs/                # 配置文件
│   └── default.yaml        # 默认配置
├── data/                   # 数据集目录
├── goalflow/
│   ├── config.py           # 配置加载器
│   ├── models/
│   │   ├── goal_flow_model.py      # 主模型
│   │   ├── perception/             # 感知模块
│   │   │   └── transfuser.py       # TransFuser BEV 特征提取
│   │   ├── goal/                   # 目标模块
│   │   │   ├── goal_vocabulary.py  # 目标候选池
│   │   │   ├── goal_scorer.py      # 目标评分器
│   │   │   ├── goal_selector.py    # 目标选择器
│   │   │   └── goal_module.py      # 目标模块
│   │   └── planning/                # 规划模块
│   │       ├── trajectory_planner.py     # 轨迹规划器
│   │       ├── trajectory_decoder.py     # 轨迹解码器
│   │       ├── trajectory_scorer.py      # 轨迹评分器
│   │       └── rectified_flow.py         # Rectified Flow
│   ├── utils/                 # 工具函数
│   │   ├── geometry.py        # 几何计算
│   │   └── metrics.py         # 评估指标
│   └── scripts/
│       ├── train.py           # 训练脚本
│       └── evaluate.py        # 评估脚本
├── pyproject.toml            # 项目配置
├── requirements.txt          # 依赖列表
└── README.md
```

## 环境配置

```bash
# 创建 conda 环境
conda create -n goalflow python=3.10
conda activate goalflow

# 安装 PyTorch
pip install torch torchvision torchaudio

# 安装项目依赖
pip install pytorch-lightning
pip install pyyaml opencv-python numpy scikit-learn matplotlib tensorboard

# 安装 nuScenes 数据集工具 (可选)
pip install nuscenes-devkit
```

## 快速开始

### 训练

```bash
python scripts/train.py --config configs/default.yaml
```

可选参数:
- `--config`: 配置文件路径 (默认: configs/default.yaml)
- `--resume`: 从检查点恢复训练
- `--gpus`: GPU 数量

### 评估

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint path/to/checkpoint
```

可选参数:
- `--config`: 配置文件路径
- `--checkpoint`: 模型检查点路径
- `--split`: 数据集划分 (val/test)
- `--visualize`: 可视化预测结果

## 模型架构

### 整体框架

GoalFlow 采用三阶段 pipeline:

1. **感知阶段**: 使用 TransFuser 提取多模态 BEV 特征
2. **目标选择阶段**: 从目标词汇表中选择最优目标点
3. **轨迹规划阶段**: 基于目标点和 BEV 特征生成候选轨迹，并进行评分排序

### 核心技术

- **TransFuser**: Transformer-based 多模态融合器，融合相机图像和点云特征
- **Goal Vocabulary**: 预定义目标点候选池，包含多种行驶场景
- **Rectified Flow**: 用于轨迹生成的 flow-based 模型

## 配置文件

主要配置项 (`configs/default.yaml`):

```yaml
model:
  perception:
    input_channels: 3
    bev_channels: 256
  goal:
    vocabulary_size: 64
  planning:
    num_trajectories: 6
    future_steps: 30

data:
  dataset: nuscenes
  data_root: /path/to/nuscenes
  batch_size: 8

training:
  max_epochs: 100
  learning_rate: 1e-4
  gpus: 1
```

## 评估指标

- **ADE (Average Displacement Error)**: 平均位移误差
- **FDE (Final Displacement Error)**: 最终位移误差
- **Miss Rate**: 未到达目标比例
- **Collision Rate**: 碰撞率

## 许可证

MIT License
