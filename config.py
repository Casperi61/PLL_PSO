# 系统参数 --------------------------------------------------
import numpy as np

# 自然频率
wn_min = 100
wn_max = 1000
# 阻尼比
zeta_min = 0.6
zeta_max = 1.2
# 鉴相器增益
Kd_min = 1.0
Kd_max = 5.0
# VCO增益
Kvco_min = 100
Kvco_max = 1000
# 时间常数
tau1_min = 1e-5
tau1_max = 1e-3
tau2_min = 1e-6
tau2_max = 1e-4

# 初始条件参数 -----------------------------------------------

# 启用随机条件
use_random_condition = True
# 测试条件数量
n_conditions = 3
# 初始频率差的随机上下界
delta_f_min = -80
delta_f_max = 80
# 初始相位差的随机上下界
delta_theta_min = -np.pi / 3
delta_theta_max = np.pi / 3
# 噪声的随机上下界
noise_level_min = 0.002
noise_level_max = 0.008

# 使用元组列表自定义初始条件, 依次是初始频率差,初始相位差,噪声.
test_conditions = [(50.0, 0.5, 0.005),
                   (20.0, 0.2, 0.004),
                   (-30.0, -0.3, 0.006)]

# 评价指标权重 ----------------------------------------------
w_lock = 100  # 收敛奖励权重
w_time = 1  # 锁定时间权重
w_error = 1  # 平均稳态误差权重
w_overshoot = 1  # 最大超调权重
w_oscillation = 1  # 相位震荡性权重

# 约束优化 --------------------------------------------------

# 是否使用约束优化
use_constrained_optimization = False
# 优化目标: time, error, overshoot, oscillation
target = "time"
# 约束条件, 优化目标设为0, 其他条件为约束条件最大值. 如果条件不限,设为float("inf")
condition_time = 0
condition_error = 0.015
condition_overshoot = float("inf")
condition_oscillation = float("inf")
