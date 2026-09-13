# MaCA Agents：Agent4 · Agent5 Pro · Agent6

基于 Multi-agent Combat Arena（MaCA）的多智能体协同对抗项目，提供三种可比较的控制策略：协同规则基线、结合规则战术的强化学习控制，以及具有预测干扰能力的独立规则策略。

项目围绕目标跟踪、搜索与追击、协同火力分配、雷达跳频和干扰展开，也提供导航学习、对战训练与成对评估入口。详细算法和接口说明见 [instruction.md](instruction.md)。

## Agent 概览

| Agent | 主要能力 | 推理依赖 | 是否需要训练权重 |
| --- | --- | --- | --- |
| [Agent4](agent/My_agent/super_agent4.py) | 目标轨迹维护、提前追击、协同齐射、弹药保留和雷达跳频 | NumPy | 否 |
| [Agent5 Pro](agent/My_agent/super_agent_5_pro.py) | 默认规则战术；支持 PPO 导航学习与搜索航向残差学习 | NumPy、PyTorch | 默认模式不需要；学习策略需要 |
| [Agent6](agent/My_agent/super_agent6.py) | 独立协同规则控制、下一步频率预测干扰、可配置战术参数 | NumPy | 否 |

- **了解规则控制**：从 Agent4 开始，代码集中在单个文件中。
- **运行独立策略或研究干扰机制**：使用 Agent6，可通过 `jam_advance=0` 关闭频率预测。
- **研究学习控制**：使用 Agent5 Pro 的导航或对战训练入口。

Agent5 Pro 的 `Agent()` 默认执行内置规则战术，模型权重不会自动参与决策。Agent6 无须导入 Agent4、Agent5 Pro 或加载模型。上述依赖指策略推理；运行完整对局仍需要 MaCA 平台。

## 项目结构

下面列出使用这三个 agent 时的主要文件：

```text
.
├── README.md
├── instruction.md                 # 策略、接口与最小对战示例
├── environment-maca.yml           # Conda 环境配置
├── requirements-maca.txt          # Python 依赖
├── agent/
│   ├── __init__.py
│   └── My_agent/
│       ├── super_agent4.py        # 协同规则基线
│       ├── super_agent_5_pro.py    # 推理、训练、评估及内置自测
│       ├── super_agent6.py        # 独立预测干扰策略
│       ├── evaluate_agent6.py     # Agent6 / Agent4 成对评估
│       └── test_agent6.py         # Agent6 单元测试
├── configuration/                # 平台与奖励配置
├── environment/                  # MaCA 引擎、原生运行库和渲染资源
└── maps/
    ├── 1000_1000_fighter10v10.map
    └── 1000_1000_2_10_vs_2_10.map
```

训练权重和运行报告由相应命令生成，不是默认规则策略的前置条件。

## 安装

### 环境要求

当前依赖配置采用以下版本，已在本地 Windows / Python 3.7 环境完成策略自测和原生对战验证：

| 组件 | 版本 |
| --- | --- |
| Python | 3.7.12 |
| NumPy | 1.21.6 |
| pandas | 1.3.5 |
| pygame | 2.5.2 |
| PyTorch | 1.13.1 |

MaCA 使用旧版 PyArmor 原生运行时。其他 Python 版本以及 Linux、macOS 的兼容性尚未在本项目中重新验证，建议先使用上述环境复现。

### 创建环境

克隆本仓库或下载并解压源码，进入包含 `README.md` 的项目根目录，然后执行：

```bash
conda env create -f environment-maca.yml
conda activate maca
```

如果已有独立的 Python 3.7 环境，也可以使用：

```bash
python -m pip install -r requirements-maca.txt
```

后续所有命令均从项目根目录执行。原生对战需要完整的 `environment/`，包括平台对应的 `_pytransform` 动态库、随平台提供的运行时文件，以及渲染资源；仅安装 pip 依赖不能替代这些文件。

## 快速开始

### 1. 验证安装

```bash
python -m unittest agent.My_agent.test_agent6
python agent/My_agent/super_agent_5_pro.py self-test --native
```

第一条检查 Agent6 的独立导入、状态复位和动作约束，第二条检查 Agent5 Pro 的策略、训练状态与原生地图交互。正常完成时会显示 `OK`。

### 2. 运行 Agent6 对 Agent4 的对战评估

先运行一组固定布局、交换双方位置的对局：

```bash
python agent/My_agent/evaluate_agent6.py --seeds 11 --layout fixed --steps 1000 --out tmp/agent6_quickstart.json
```

该命令执行 2 场对局，终端输出汇总，并自动创建输出目录，将逐局记录写入 `tmp/agent6_quickstart.json`。无需预训练权重。

进一步评估固定和随机布局：

```bash
python agent/My_agent/evaluate_agent6.py --seeds 11 23 37 --layout both --steps 1000 --out tmp/agent6_evaluation.json
```

共执行 `3 个种子 × 2 种布局 × 2 个位置 = 12 场`。报告包含胜负平、胜率、存活差、逐局结果、种子、地图与源码摘要，以及运行库版本。

如需直接接入环境、替换任一方 agent 或开启窗口渲染，请使用 [最小对战示例](instruction.md#最小对战示例)。仓库中的旧 `fight.py` / `fight_mp.py`（若保留）使用旧策略路径约定，不作为这里三个 agent 的统一运行入口。

## 接口与参数

三个 agent 都提供 `Agent` 类，使用 MaCA 的 `raw` 观测：

```python
from agent.My_agent.super_agent6 import Agent

agent = Agent(jam_advance=7)
agent.set_map_info(size_x, size_y, detector_num, fighter_num)
detector_actions, fighter_actions = agent.get_action(obs_dict, step_cnt)
```

以上为接入片段，`size_x`、单位数量和 `obs_dict` 应从环境获取。每局先设置地图信息，再按递增步数调用 `get_action`。

| 输出 | 形状 | 含义 |
| --- | --- | --- |
| `detector_actions` | `(侦察机数量, 2)` | 航向、雷达频率 |
| `fighter_actions` | `(战斗机数量, 4)` | 航向、雷达频率、干扰频率、攻击编码 |

输出为 `np.int32` 数组。Agent4 使用双方单位数量对称的编码假设；Agent5 Pro 和 Agent6 可通过 `set_map_info(..., enemy_unit_count=N)` 指定敌方总数。动作范围和完整调用顺序见 [接口说明](instruction.md#统一接口)。

Agent6 默认 `jam_advance=7`，对应每步频率加 7 的对手跳频规律。它是一种针对特定规律的策略，并非对任意对手都有效。可通过以下实例配置进行比较：

```python
baseline = Agent(jam_advance=0)
predictive = Agent(jam_advance=7)
```

## Agent5 Pro：训练与推理

### 控制模式

| 接口或模式 | 用途 |
| --- | --- |
| `Agent()` / `combat_control="agent4"` | 默认规则对战，不依赖训练权重 |
| `Agent(..., combat_control="residual")` | 加载已训练的 residual 权重，在搜索阶段小幅修正规则航向 |
| `Agent(..., combat_control="legacy")` | 旧版网络绝对航向控制，供实验比较 |
| `NavigationAgent()` | 纯航向控制，无权重时使用确定性航路点引导 |

残差模式只在没有目标轨迹且战斗机仍有弹药时调整搜索航向；目标追击和武器通道仍由规则负责。导航模式的雷达、干扰和攻击通道保持为零。

### 导航训练

```bash
python agent/My_agent/super_agent_5_pro.py train --backend maca --updates 300 --out tmp/super5_navigation
python agent/My_agent/super_agent_5_pro.py evaluate --checkpoint tmp/super5_navigation/best.pt
```

导航训练使用无弹药处理后的地图。若只检查学习流程，可将训练后端设为 `--backend toy`，该后端不需要加载原生 MaCA 环境。

### 对战残差训练

```bash
python agent/My_agent/super_agent_5_pro.py combat-train --combat-control residual --updates 300 --out tmp/super5_combat
python agent/My_agent/super_agent_5_pro.py combat-evaluate --combat-control residual --checkpoint tmp/super5_combat/best.pt --seeds 11 23 37 --out tmp/super5_combat_evaluation.json
```

默认对手为 Agent4。对战评估在相同种子下交换双方位置，并包含 Agent4 基线对照；添加 `--random-pos` 可评估随机初始布局。

训练目录包含：

| 文件 | 用途 |
| --- | --- |
| `best.pt` | 按训练过程中的评估分数保存的最佳检查点 |
| `latest.pt` | 恢复训练使用的最新检查点 |
| `metrics.jsonl` | 配置、更新和评估记录 |

在原目录继续训练 100 次更新：

```bash
python agent/My_agent/super_agent_5_pro.py combat-train --resume tmp/super5_combat/latest.pt --updates 100 --out tmp/super5_combat
```

新实验应选择空的输出目录，已有结果的目录需要使用 `--resume`。更多参数可通过 `python agent/My_agent/super_agent_5_pro.py --help` 或相应子命令的 `--help` 查看。

### 权重使用注意

`residual` 推理要求对应类型且已完成训练更新的权重，导航权重不能直接用作 residual 对战权重。若仓库附带 `super5_runs/default/` 的历史权重，其配套报告标记为 `nonweapon_navigation`，应按导航用途使用。

默认 `Agent()` 不会自动读取默认目录或 `SUPER5_CHECKPOINT` 环境变量；仅指定权重路径也不会把默认规则模式切换成学习模式。需要显式选择：

```python
from agent.My_agent.super_agent_5_pro import Agent

agent = Agent(
    checkpoint="tmp/super5_combat/best.pt",
    combat_control="residual",
    device="cpu",
)
```

## 实验复现

对比策略时，应保持地图、种子、布局、最大步数和对手一致，并交换双方位置。调整参数后使用另一组种子验证，避免仅依据调参对局判断效果。

Agent4 自对战基线：

```bash
python agent/My_agent/evaluate_agent6.py --baseline --seeds 11 23 37 --layout both --steps 1000 --out tmp/agent4_selfplay.json
```

Agent5 Pro 默认规则模式评估：

```bash
python agent/My_agent/super_agent_5_pro.py combat-evaluate --combat-control agent4 --seeds 11 23 37 --out tmp/super5_rules.json
```

Agent6 的评估脚本复用了 Agent5 Pro 中的环境包装，因此该评估入口需要 PyTorch；Agent6 本身的推理不需要 PyTorch。

2026-09-13 的本地验证中，Agent6 的 6 项单元测试、Agent5 Pro 的 20 项原生自测均通过，最小对战示例运行至结束。这些检查验证接口与实现行为，不代表某个策略在所有地图或对手上占优，也不等同于全新机器上的安装验证。

## 常见问题

| 问题 | 处理方式 |
| --- | --- |
| 找不到 `interface` | 从项目根目录运行本文入口；自定义脚本需要将 `environment/` 加入模块搜索路径，参照最小示例 |
| NumPy / PyTorch DLL 加载失败 | 确认已激活 `maca` 环境，避免混用系统 Python 与 Conda 环境 |
| PyArmor / `_pytransform` 加载失败 | 检查平台运行库是否完整，并核对操作系统、Python 版本与位数 |
| residual 检查点被拒绝 | 使用 `combat-train --combat-control residual` 生成对应权重 |
| 输出目录已有训练结果 | 选择新目录，或从 `latest.pt` 使用 `--resume` 续训 |
| 旧入口找不到 `fix_rule` 或 `surper_agent*` | 使用本文评估入口或 instruction 中的最小对战示例 |

## 来源与进一步阅读

原始 MaCA 平台由 CETC-TFAI 团队开发：[CETC-TFAI/MaCA](https://github.com/CETC-TFAI/MaCA)。本项目在其环境和接口上开展自定义 agent 实验，保留原平台源码中的作者说明。

Agent5 Pro 的多层反事实信用分配属于论文启发的工程实现，采用固定基线权重，未实现 CMA-ES，也不声明复现原论文全部实验。

策略流程、参数默认值、动作编码、回放和完整对战示例见 [instruction.md](instruction.md)。
