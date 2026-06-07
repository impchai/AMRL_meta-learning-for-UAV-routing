# UAV-IoT PAoI Optimization with AMRL

本项目面向无人机辅助物联网（UAV-assisted IoT）场景，研究无人机在多个地面物联网节点之间进行数据采集时的路径规划与信息新鲜度优化问题。代码基于 PyTorch 实现了一个马尔可夫决策过程（MDP）环境，并比较了自适应元强化学习（AMRL）、标准强化学习、启发式策略和 TSP 类路径策略在最大峰值信息年龄（Peak Age of Information, PAoI）指标上的表现。

仓库当前包含：

- `train_denoise_optimized_AMRL.py`：完整实验脚本，包含环境建模、智能体定义、训练、评估和可视化。


## 项目特点

- 构建 UAV 辅助 IoT 数据采集场景的 MDP 环境。
- 使用 PAoI 作为核心性能指标，优化多节点数据采集顺序。
- 实现 AMRL（Attention-based Meta Reinforcement Learning）方法，用于提升跨随机拓扑任务的快速适应能力。
- 提供 Standard RL、Random Walk、Max-Age First、Exact Optimal TSP、Fast Greedy TSP 等对比方法。
- 自动输出轨迹图、PAoI 锯齿变化图、性能柱状图和 JSON 结果文件。

## 系统模型

### 网络场景

系统由一个基站、一架无人机和 `K` 个地面物联网节点组成。基站位于二维区域中心，物联网节点在给定区域内随机分布。无人机从基站出发，以固定高度飞行，依次访问各物联网节点完成数据同步，最后返回基站。

默认环境参数包括：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `K` | 物联网节点数量 | 由实验场景指定，默认 `[4, 8, 12, 20]` |
| `area_size` | 方形区域边长 | `1000.0 m` |
| `H` | 无人机飞行高度 | `100.0 m` |
| `V` | 无人机飞行速度 | `20.0 m/s` |
| `B` | 通信带宽 | `1 MHz` |
| `P_tx` | 节点发射功率 | `0.1 W` |
| `T_sync` | 固定同步时间 | `0.5 s` |
| `data_arrival_rates` | 节点数据到达率 | 随机采样于 `[5e4, 2e5]` |

### 状态空间

环境状态由以下信息拼接得到：

- 无人机当前位置；
- 基站和所有 IoT 节点的归一化二维坐标；
- 无人机到各节点和基站的归一化距离；
- 各节点当前缓存队列长度；
- 各节点当前 PAoI；
- 各节点是否已经被访问。

状态维度定义为：

```text
state_dim = 3 + (K + 1) * 2 + (K + 1) + K * 3
```

其中 `K + 1` 表示 `K` 个 IoT 节点加 1 个基站。

### 动作空间

动作空间大小为：

```text
action_space_n = K + 1
```

动作含义如下：

- `0`：返回基站；
- `1 ~ K`：访问对应编号的 IoT 节点。

环境使用动作掩码限制非法动作：

- 在所有 IoT 节点访问完成前，不允许返回基站；
- 已访问过的 IoT 节点不允许重复访问。

### 状态转移

每次动作会触发以下过程：

1. 计算无人机从当前位置到目标节点的飞行时间；
2. 若目标为 IoT 节点，则根据信道模型和队列长度计算悬停同步时间；
3. 更新全局时间、所有节点队列长度和 PAoI；
4. 若访问 IoT 节点，则清空该节点队列，并重置该节点 PAoI；
5. 当所有节点访问完成并返回基站时，当前 episode 结束。

### 奖励函数

奖励函数以降低最大 PAoI 为目标，同时加入飞行距离和悬停时间惩罚：

```text
reward = -max(PAoI) - distance_penalty - 0.05 * T_hov
```

当所有节点访问完成且无人机返回基站时：

```text
reward = 500.0 - max(PAoI)
```

为稳定训练，脚本中将奖励整体缩放为原来的 `1/100`。

## 算法说明

### AMRL

AMRL 使用基于注意力机制的 Actor-Critic 网络：

- `Attention_ActorCritic` 根据无人机位置和节点位置计算注意力分数；
- actor 输出各动作 logits；
- critic 输出当前状态价值；
- 内环使用少量轨迹进行快速任务适应；
- 外环进行一阶元学习更新。

### Standard RL

Standard RL 使用普通 MLP Actor-Critic 网络：

- 两层全连接隐藏层；
- actor 输出动作分布；
- critic 估计状态价值；
- 评估前可在新任务上执行少量 fine-tuning。

### 对比方法

- `RandomAgent`：在合法动作中随机选择；
- `MaxPAoIAgent`：优先访问 PAoI 和队列压力更高的节点；
- `OptimalTSPAgent`：对小规模场景枚举所有路径，求精确 TSP 顺序；
- `FastTSPAgent`：对大规模场景使用最近邻贪心近似 TSP。

## 输入与输出

### 输入

项目不依赖外部数据集。每次实验会自动随机生成：

- IoT 节点位置；
- 节点数据到达率；
- 不同节点规模 `K` 对应的测试拓扑。

默认实验配置位于 `train_denoise_optimized_AMRL.py` 的主程序入口：

```python
TRAIN_ITERATIONS = 8000
K_scenarios = [4, 8, 12, 20]
```

如果只想快速验证代码是否能跑通，可以将其临时改为：

```python
TRAIN_ITERATIONS = 10
K_scenarios = [4]
```

### 输出

运行完成后，结果默认保存在：

```text
results_K_scenarios/
```

主要输出包括：

| 文件 | 含义 |
| --- | --- |
| `paoi_sawtooth_AMRL_K*.png` | AMRL 策略下各节点 PAoI 随时间变化的锯齿图 |
| `trajectories_K*.png` | 不同算法的无人机轨迹对比图 |
| `bar_chart_comparison_K*.png` | 不同算法平均最大 PAoI 对比柱状图 |
| `uav_routing_results.json` | 各节点规模下不同算法的数值结果 |

JSON 输出示例结构：

```json
{
  "K_4": {
    "node_count": 4,
    "evaluation_results": {
      "Random Walk": 0.0,
      "Max-Age First (MAF)": 0.0,
      "Standard RL (Fine-tuned)": 0.0,
      "AMRL": 0.0,
      "Exact Optimal TSP": 0.0
    }
  }
}
```

实际数值会随训练过程、随机拓扑和硬件环境有所变化。

## 环境依赖

建议使用 Python 3.9 或更高版本。

核心依赖：

```text
numpy
torch
matplotlib
```

安装示例：

```bash
pip install numpy torch matplotlib
```

如果使用 Conda：

```bash
conda create -n uav-paoi python=3.10
conda activate uav-paoi
pip install numpy torch matplotlib
```

## 使用方法

1. 克隆或下载本仓库：

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

2. 安装依赖：

```bash
pip install numpy torch matplotlib
```

3. 运行完整实验：

```bash
python train_denoise_optimized_AMRL.py
```

完整默认实验包含多个节点规模，并且每个场景训练 8000 轮，运行时间可能较长。建议首次运行时先降低 `TRAIN_ITERATIONS`。

4. 查看结果：

```text
results_K_scenarios/
```

## 代码结构

```text
.
├── README.md
└── train_denoise_optimized_AMRL.py
```

脚本内部结构如下：

- `UAV_IoT_Env`：UAV-IoT MDP 环境；
- `MLP_ActorCritic`：标准强化学习网络；
- `Attention_ActorCritic`：AMRL 使用的注意力 Actor-Critic 网络；
- `AMRL`：一阶元强化学习训练流程；
- `Standard_RL`：标准 Actor-Critic 训练流程；
- `RandomAgent`、`MaxPAoIAgent`、`OptimalTSPAgent`、`FastTSPAgent`：对比策略；
- `evaluate_agents`：多智能体评估；
- `plot_paoi_sawtooth`、`plot_comprehensive_comparison`：结果可视化。

## 复现实验

脚本中设置了随机种子：

```python
torch.manual_seed(2024)
np.random.seed(2024)
```

评估阶段也固定了随机种子，以便在相同运行环境下尽量复现实验结果。由于 PyTorch 后端、CPU/GPU、库版本等因素可能存在差异，结果可能出现轻微浮动。

