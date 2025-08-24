# 自适应PLL参数优化平台（基于PSO）

本项目是一个灵活可复用的**二阶锁相环（PLL）系统优化平台**，集成了高精度仿真模型与改进的自适应粒子群算法（PSO），支持多条件性能评估、约束优化、多目标可调评分机制，适用于控制理论研究、算法工程开发及科研论文实验验证。

---

## 📌 项目特色

- 🧠 **二阶PLL仿真系统**：支持相位误差演化模拟、非线性动态建模、初始偏频与随机噪声扰动。
- 🐝 **自适应PSO优化器**：融合惯性调整、自适应速度机制、多次重启与最优保持等机制。
- 🔁 **多初始条件评估机制**：支持多组初始条件下的综合打分与鲁棒性测试。
- ⚖️ **多维性能指标集成**：
  - 锁定时间
  - 稳态误差
  - 过冲幅度
  - 相位振荡性
- 📊 **权重可调的评分函数**：支持按需权衡不同指标，或指定目标指标最优化、其余作为约束。
- 🧪 **参数鲁棒性分析工具**：输出最优解分布、均值、标准差、变异系数（CV）等统计量。
- 🔧 **配置解耦**：所有参数集中于 `config.py`，无需修改核心代码。

---

## 📁 项目结构

```bash
project/
│
├── main.py             # 主程序入口，运行全流程优化实验
├── pll_env.py          # 锁相环仿真环境类（PLLEnvironment）
├── optimizer.py        # 自适应粒子群优化器（PSO）
├── config.py           # 所有仿真与优化配置
└── README.md           # 项目说明文档（即本文）
```

## 🚀 快速开始

1. 克隆仓库


```
git clone https://github.com/Casperi61/PLL_PSO.git

cd PLL_PSO
```

2. 安装依赖

```
pip install numpy=2.3.2 scipy=1.16.0
```

3. 运行示例实验

```
python main.py
```

## ⚙️ 配置说明（config.py）

在config.py中配置系统参数,初始条件,评价指标权重,约束优化参数

### 系统参数
配置自然频率,阻尼比,鉴相器增益,VCO增益,时间常数等系统参数等上下界.
如果需要确定的系统参数,请将上下界设为相同值.

### 初始条件参数
你可以使用随机的初始条件(use_random_condition = True),并设置条件数量和随机条件的上下界.
您也可以自定义初始条件(use_random_condition = False),并使用元组列表在test_conditions自定义初始条件, 依次是初始频率差,初始相位差,噪声.比如:
```python
test_conditions = [(50.0, 0.5, 0.005),
                   (20.0, 0.2, 0.004),
                   (-30.0, -0.3, 0.006)]
```
### 评价指标函数
设置收敛奖励权重,锁定时间权重,平均稳态误差权重,最大超调权重,相位震荡性权重.

### 约束优化参数
use_constrained_optimization = True 以启用带有约束条件的优化过程.
使用target明确优化目标,可优化目标有time, error, overshoot, oscillation.
并设置约束条件.比如:
```python
# 优化目标: time, error, overshoot, oscillation
target = "time"
# 约束条件, 优化目标设为0, 其他条件为约束条件最大值. 如果条件不限,设为float("inf")
condition_time = 0
condition_error = 0.015
condition_overshoot = float("inf")
condition_oscillation = float("inf")
```


