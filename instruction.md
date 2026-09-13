# Agent4、Agent5 Pro、Agent6 使用说明

本文对应 `agent/My_agent/super_agent4.py`、`super_agent_5_pro.py`、`super_agent6.py` 的当前实现。所有命令从仓库根目录执行；环境安装和上传清单见 [README.md](README.md)。

## 统一接口

```python
agent = Agent()
agent.get_obs_ind()  # 返回 "raw"
agent.set_map_info(size_x, size_y, detector_num, fighter_num)
detector_actions, fighter_actions = agent.get_action(obs_dict, step_cnt)
```

输入为平台原始观测，包含 `detector_obs_list`、`fighter_obs_list` 和 `joint_obs_dict`，以及单位存活状态、位置、航向、可见目标、接收信号和剩余弹药等字段。策略只使用传入的观测，不读取敌方隐藏状态。

| 输出 | 形状 | 列含义 |
| --- | --- | --- |
| `detector_actions` | `(detector_num, 2)` | 航向、雷达频率 |
| `fighter_actions` | `(fighter_num, 4)` | 航向、雷达频率、干扰频率、攻击编码 |

输出为 `np.int32`，阵亡单位保留对应行并输出零动作。航向为 0–359；雷达 0 为关闭、1–10 为频率；干扰 0 为关闭、1–10 为定频、11 为宽带。雷达动作不是扫描角度。

攻击编码：0 不攻击，`1..N` 表示长程弹目标 ID，`N+1..2N` 表示短程弹目标 ID 加敌方总单位数 `N`。Agent4 用己方总数作为偏移，因此本文示例使用双方数量对称的自带地图。Agent5 Pro 和 Agent6 的 `set_map_info` 可额外传入 `enemy_unit_count=N`。

每局从第 1 步开始递增调用。不要在同一步重复调用有状态策略：步数回退或重复会触发新局状态处理。新局重新调用 `set_map_info` 可明确清空历史。

## Agent4：协同规则基线

依赖 NumPy 和标准库，无训练、无权重。

1. 合并雷达可见目标与联合被动探测，维护最多 25 步的目标轨迹；估计速度并限制外推，清除已报告阵亡目标。
2. 根据距离、已分配追踪者数量和观测时效分配追击目标，提前预测位置。无目标时先集结、再分区巡航。
3. 仅从本机雷达可见列表选择射击目标，用单机发射间隔和全队待结算齐射数量限制重复攻击。
4. 保留最后一枚短程弹，近距离危险或后期释放。无弹战斗机近距离脱离，继续提供探测。
5. 战斗机雷达按 `1 + (step * 7 + index * 3) % 10` 跳频；接收信号有多数频率时定频干扰，否则宽带干扰。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `long_range` / `short_range` | 120 / 50 | 策略射击距离阈值 |
| `fire_interval` | 2 | 单机两次发射最小步数间隔 |
| `salvo_cap` / `shot_wait` | 2 / 8 | 单目标待结算射击上限和保留步数 |
| `formation_width` | 100 | 初期集结展开宽度 |
| `lead_steps` | 4 | 运动预测提前量 |
| `reserve_release_step` | 600 | 解除短程弹保留的步数 |

这些是经验策略参数，不是环境真实射程或命中率的声明。Agent4 构造函数不接收参数，可在创建实例后修改同名属性。

## Agent5 Pro：规则战术与学习航向

该文件包含推理接口、环境包装、训练器、评估器和自测。即使使用默认规则模式，导入时仍需要 PyTorch。

### 对战模式

| 模式 | 创建方式 | 行为 |
| --- | --- | --- |
| `agent4`（默认） | `Agent()` | 使用文件内置 `Agent4Tactics`，不需要权重 |
| `residual` | `Agent(checkpoint="...", combat_control="residual")` | 规则控制战术与武器，网络修正搜索航向 |
| `legacy` | `Agent(checkpoint="...", combat_control="legacy")` | 网络输出绝对航向，用于旧模式实验 |

残差仅在没有目标轨迹且战斗机仍有弹药时生效；侦察机保留规则控制。实际离散修正为 0、-12 到 -1、1 到 11 度。`residual` 必须加载至少完成一次更新的 residual 类型权重，否则报错。

非默认模式按显式 `checkpoint`、环境变量 `SUPER5_CHECKPOINT`、`super5_runs/default/best.pt` 的顺序寻找权重。默认 `Agent()` 不读取后两者。即使显式传入权重，`agent4` 模式仍使用规则动作；无权重的 `legacy` 是未训练网络，不应当作训练成果。

### 导航与学习方法

`NavigationAgent` 只输出航向，其他通道为零。没有权重时使用确定性的航路点引导；权重需要显式指定，不读取环境变量或默认目录。调用 `set_navigation_goals(goals_xy)` 设置目标，坐标使用地图单位，顺序为侦察机在前、战斗机在后。

学习部分使用共享 Actor、带注意力的集中式 Critic 和 PPO，结合联合、个体和相关集合三个层次的反事实基线。基线权重固定为 0.5 / 0.25 / 0.25。这是论文启发的实现，不是论文完整复现，也没有实现 CMA-ES。

导航训练和评估：

```bash
python agent/My_agent/super_agent_5_pro.py train --backend maca --updates 300 --out tmp/super5_navigation
python agent/My_agent/super_agent_5_pro.py evaluate --checkpoint tmp/super5_navigation/best.pt
python agent/My_agent/super_agent_5_pro.py navigation-evaluate --backend maca --seeds 200003 200009 200017
```

`train` / `evaluate` 使用导航任务，对原生地图进行无弹药处理。`toy` 后端用于不加载原生 MaCA 的导航流程检查，不能代替对战验证。

对战训练和评估：

```bash
python agent/My_agent/super_agent_5_pro.py combat-train --combat-control residual --updates 300 --out tmp/super5_combat
python agent/My_agent/super_agent_5_pro.py combat-evaluate --combat-control residual --checkpoint tmp/super5_combat/best.pt --seeds 11 23 37 --out tmp/super5_combat_eval.json
python agent/My_agent/super_agent_5_pro.py combat-evaluate --combat-control agent4 --seeds 11 23 37 --out tmp/super5_rules_eval.json
```

对战默认使用 `adversarial` 模式，对手来自 `super_agent4.py`，因此必须保留该文件。对战评估用相同种子、交换双方位置，分别评估 Agent5 候选和 Agent4 基线。默认固定布局，添加 `--random-pos` 可验证随机布局。

训练输出 `best.pt`、`latest.pt`、`metrics.jsonl`。`latest.pt` 保存续训状态，`--updates` 表示额外更新次数：

```bash
python agent/My_agent/super_agent_5_pro.py combat-train --resume tmp/super5_combat/latest.pt --updates 100 --out tmp/super5_combat
```

现有 `super5_runs/default/validation_report.json` 将实验标记为 `nonweapon_navigation`。该目录现有权重应按导航成果说明，不能直接当作 residual 对战权重或战斗能力提升的证据。发布学习策略时，应同时记录权重、地图、种子、控制模式和验证报告。

## Agent6：预测干扰的独立规则策略

Agent6 将追踪、搜索、齐射和弹药控制完整放在单个文件中，只需 NumPy，不导入其他 agent，也没有训练过程或模型文件。

主要变化是根据收到的频率预测下一步干扰频率。默认 `jam_advance=7`，对应 Agent4 每步加 7 的跳频规律；没有明确多数频率时仍使用宽带干扰。该规则针对特定跳频规律，不代表能预测任意对手。

```python
from agent.My_agent.super_agent6 import Agent

agent = Agent()
without_prediction = Agent(jam_advance=0)
experimental = Agent(jam_advance=7, standoff=90, salvo_cap=2)
```

`standoff` 默认 0，关闭额外距离控制。开启后，有目标轨迹且仍有长程弹时，距离过近则退离。构造函数也接受 Agent4 参数表中的参数，并检查参数名、类型和范围；`jam_advance` 范围为 0–9。输出经过形状、类型和动作编码边界检查。

## 最小对战示例

将以下代码保存为仓库根目录的 `run_match.py`，执行 `python run_match.py`。这是供复制的示例，本次未另行创建该文件。它不依赖旧 agent 或 `fight.py`。

```python
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
os.chdir(str(root))
sys.path.insert(0, str(root / "environment"))
os.environ["PATH"] = str(Path(sys.prefix) / "Library" / "bin") + os.pathsep + os.environ.get("PATH", "")

from interface import Environment
from agent.My_agent.super_agent4 import Agent as Agent4
from agent.My_agent.super_agent_5_pro import Agent as Agent5
from agent.My_agent.super_agent6 import Agent as Agent6

left, right = Agent4(), Agent6()  # 可替换为 Agent5()
env = Environment(
    "maps/1000_1000_fighter10v10.map", "raw", "raw",
    max_step=1000, render=False, random_pos=False, random_seed=11,
    log=False,
)
size_x, size_y = env.get_map_size()
d1, f1, d2, f2 = env.get_unit_num()
left.set_map_info(size_x, size_y, d1, f1)
right.set_map_info(size_x, size_y, d2, f2)

for step in range(1, 1001):
    obs1, obs2 = env.get_obs()
    a1d, a1f = left.get_action(obs1, step)
    a2d, a2f = right.get_action(obs2, step)
    env.step(a1d, a1f, a2d, a2f)
    if env.get_done():
        break

reward = env.get_reward()
print("steps:", step)
print("side1 game reward:", reward[2], "side2 game reward:", reward[5])
print("winner:", "side1" if reward[2] > reward[5] else "side2" if reward[2] < reward[5] else "draw")
```

需要窗口时改为 `render=True`。记录回放时先创建 `log/` 目录，再将 `log=False` 改为 `log="agent4_vs_agent6"`。保留 `replay.py` 时，可在 PowerShell 中回放：

```powershell
$env:PYTHONPATH = "$PWD\environment;$env:PYTHONPATH"
python replay.py agent4_vs_agent6
```

## 自测与评估

```bash
python -m unittest agent.My_agent.test_agent6
python agent/My_agent/super_agent_5_pro.py self-test
python agent/My_agent/super_agent_5_pro.py self-test --native
python agent/My_agent/evaluate_agent6.py --seeds 11 23 37 --layout both --steps 1000 --out tmp/agent6_eval.json
python agent/My_agent/evaluate_agent6.py --baseline --seeds 11 23 37 --layout both --steps 1000 --out tmp/agent4_selfplay.json
```

Agent6 评估复用 Agent5 Pro 的环境包装，因此评估需要 PyTorch，而 Agent6 本身推理不需要。`--baseline` 是 Agent4 对 Agent4。

报告记录胜负平、存活差、种子、地图和源码摘要、运行库版本及逐局结果。`--sweep` 和 `tune_agent6_ranges.py` 用于开发调参，之后应使用不同种子重新验证。本文不承诺任何版本在所有地图和对手上更强，已有历史报告不等同于当前代码的新测试结果。

## 常见问题

- `No module named interface`：从根目录运行，并将 `environment/` 加入模块搜索路径；最小示例已处理。
- NumPy / PyTorch DLL 加载失败：激活依赖文件对应环境，避免混用系统 Python 与 Conda 环境。
- PyArmor 或 `_pytransform` 加载失败：核对完整运行包、平台原生库和 Python 版本，不要只复制 `.py`。
- residual 权重校验失败：使用 `combat-train --combat-control residual` 生成匹配权重，不要传入导航或 legacy 权重。
- 找不到 `surper_agent*` / `fix_rule`：旧脚本依赖已删除的旧路径，改用本文示例和保留的评估脚本。

## 本次文档核验

2026-09-13 在本地 Windows / 项目 `.conda` Python 3.7 环境完成：

- 文档内 Python 示例语法检查通过，最小 Agent4 对 Agent6 对战实际运行至结束。
- Agent6 单元测试 6 项通过。
- Agent5 Pro `self-test --native` 20 项通过，包含原生地图测试。

以上验证使用当前本地环境，不等于已经在全新机器上验证精简上传包；未重新进行长时间训练或跨平台验证。
