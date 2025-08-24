from idlelib.grep import walk_error
import config as cf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


class PLLEnvironment:
    """改进的二阶锁相环仿真环境"""

    def __init__(self):
        self.param_ranges = {
            'wn': (cf.wn_min, cf.wn_max),  # 自然频率
            'zeta': (cf.zeta_min, cf.zeta_max),  # 阻尼比
            'Kd': (cf.Kd_min, cf.Kd_max),  # 鉴相器增益
            'Kvco': (cf.Kvco_min, cf.Kvco_max),  # VCO增益
            'tau1': (cf.tau1_min, cf.tau1_max),  # 时间常数1
            'tau2': (cf.tau2_min, cf.tau2_max)  # 时间常数2
        }

        # 初始条件范围
        self.delta_f_range = (-100, 100)  # 频率偏差 Hz
        self.delta_theta_range = (-np.pi / 4, np.pi / 4)  # 相位偏差 rad

        # 仿真参数
        self.t_sim = 0.01  # 仿真时长
        self.dt = 1e-6  # 时间步长

    def reset(self, delta_f=None, delta_theta=None, noise_level=None):
        """重置环境条件"""
        self.delta_f0 = delta_f if delta_f is not None else np.random.uniform(*self.delta_f_range)
        self.delta_theta0 = delta_theta if delta_theta is not None else np.random.uniform(*self.delta_theta_range)
        self.noise_level = noise_level if noise_level is not None else np.random.uniform(0.001, 0.01)

        return np.array([self.delta_f0, self.delta_theta0, self.noise_level])

    def pll_dynamics(self, t, y, params):
        """PLL动态方程"""
        theta_e, omega_e = y
        wn, zeta, Kd, Kvco, tau1, tau2 = params

        # 环路增益
        K = Kd * Kvco

        # 相位检测器输出（非线性）
        phase_det_out = np.sin(theta_e)

        # 添加噪声
        noise = self.noise_level * np.random.normal(0, 0.1) if hasattr(self, 'noise_level') else 0

        # 二阶系统动态方程
        dtheta_dt = omega_e
        domega_dt = -2 * zeta * wn * omega_e - wn ** 2 * theta_e + K * phase_det_out + noise

        return [dtheta_dt, domega_dt]

    def simulate_pll(self, params):
        """执行PLL仿真"""
        try:
            # 初始条件
            theta_e0 = self.delta_theta0
            omega_e0 = 2 * np.pi * self.delta_f0
            y0 = [theta_e0, omega_e0]

            # 时间序列
            t_span = (0, self.t_sim)
            t_eval = np.arange(0, self.t_sim, self.dt)

            # 求解微分方程
            sol = solve_ivp(
                lambda t, y: self.pll_dynamics(t, y, params),
                t_span, y0, t_eval=t_eval,
                method='RK45', rtol=1e-6, atol=1e-8
            )

            if not sol.success:
                return None, None, None, False

            theta_e = sol.y[0]
            omega_e = sol.y[1]
            t = sol.t

            # 检查数值稳定性
            if np.any(np.abs(theta_e) > 10) or np.any(np.abs(omega_e) > 1e6):
                return None, None, None, False

            return theta_e, omega_e, t, True

        except Exception as e:
            return None, None, None, False

    def evaluate_performance(self, params, return_details=False):
        """评估PLL性能"""
        theta_e, omega_e, t, success = self.simulate_pll(params)

        if not success:
            if return_details:
                return -1000, {}
            return -1000

        # 性能指标计算
        lock_threshold = 0.1  # 锁定阈值
        phase_error = np.abs(theta_e)

        # 1. 锁定时间
        locked_indices = np.where(phase_error < lock_threshold)[0]
        if len(locked_indices) > len(theta_e) * 0.2:
            lock_time = locked_indices[0] * self.dt
            lock_success = 1
        else:
            lock_time = self.t_sim
            lock_success = 0

        # 2. 稳态误差
        if lock_success:
            steady_state_error = np.mean(phase_error[locked_indices])
        else:
            steady_state_error = np.mean(phase_error)

        # 3. 过冲
        max_error = np.max(phase_error)
        # 参考稳态误差作为基线
        overshoot = max(0, max_error - steady_state_error)

        # 4. 平均误差
        avg_error = np.mean(phase_error)

        # 5. 稳定性（振荡检测）
        if lock_success and len(theta_e) > 100:
            steady_window = theta_e[-100:]
            oscillation = np.std(steady_window)
        else:
            oscillation = np.std(theta_e)

        if not cf.use_constrained_optimization:

            # 综合性能评分
            w_lock = cf.w_lock
            w_time = cf.w_time
            w_error = cf.w_error
            w_overshoot = cf.w_overshoot
            w_oscillation = cf.w_oscillation
            score = (
                    w_lock * lock_success +
                    max(0, 50 - w_time * lock_time * 1000) +
                    max(0, 50 - w_error * avg_error * 100) +
                    max(0, 30 - w_overshoot * overshoot * 100) +
                    max(0, 20 - w_oscillation * oscillation * 1000)
            )
        else:
            if cf.target == "time":
                if steady_state_error > cf.condition_error or overshoot > cf.condition_overshoot or oscillation > cf.condition_oscillation:
                    out_of_condition = 1
                else:
                    out_of_condition = 0
                score = (
                        cf.w_lock * lock_success +
                        max(0, 50 - cf.w_time * lock_time * 1000) -
                        out_of_condition * 100
                )
            if cf.target == "error":
                if lock_time > cf.condition_time or overshoot > cf.condition_overshoot or oscillation > cf.condition_oscillation:
                    out_of_condition = 1
                else:
                    out_of_condition = 0
                score = (
                        cf.w_lock * lock_success +
                        max(0, 50 - cf.w_error * avg_error * 100) -
                        out_of_condition * 100
                )
            if cf.target == "overshoot":
                if lock_time > cf.condition_time or steady_state_error > cf.condition_error or oscillation > cf.condition_oscillation:
                    out_of_condition = -1
                else:
                    out_of_condition = 0
                score = (
                        cf.w_lock * lock_success +
                        max(0, 30 - cf.w_overshoot * overshoot * 100) -
                        out_of_condition * 100
                )
            if cf.target == "oscillation":
                if lock_time > cf.condition_time or steady_state_error > cf.condition_error or overshoot > cf.condition_overshoot:
                    out_of_condition = -1
                else:
                    out_of_condition = 0
                score = (
                        cf.w_lock * lock_success +
                        max(0, 20 - cf.w_oscillation * oscillation * 1000) -
                        out_of_condition * 100
                )

        if return_details:
            details = {
                'lock_time': lock_time,
                'lock_success': lock_success,
                'steady_state_error': steady_state_error,
                'avg_error': avg_error,
                'overshoot': overshoot,
                'oscillation': oscillation,
                'max_error': max_error,
                'theta_e': theta_e,
                'omega_e': omega_e,
                't': t
            }
            return score, details

        return score


class AdaptivePSOOptimizer:
    """自适应粒子群优化算法 - 带详细输出"""

    def __init__(self, n_particles=30, n_dimensions=6, bounds=None):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.bounds = bounds or [(-2, 2)] * n_dimensions

        # PSO参数
        self.w = 0.7  # 惯性权重
        self.c1 = 1.5  # 个体学习因子
        self.c2 = 1.5  # 社会学习因子
        self.w_min = 0.4  # 最小惯性权重
        self.w_max = 0.9  # 最大惯性权重

        # 初始化粒子群
        self.reset()

    def reset(self):
        """重置粒子群"""
        # 初始化位置和速度
        self.positions = np.random.uniform(
            [b[0] for b in self.bounds],
            [b[1] for b in self.bounds],
            (self.n_particles, self.n_dimensions)
        )

        self.velocities = np.random.uniform(
            -0.1, 0.1, (self.n_particles, self.n_dimensions)
        )

        # 个体最优和全局最优
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.n_particles, -float('inf'))
        self.global_best_position = None
        self.global_best_score = -float('inf')

        # 历史记录
        self.score_history = []
        self.diversity_history = []
        self.detailed_history = []  # 新增：详细历史记录

    def normalize_params(self, normalized_params):
        """将归一化参数转换为实际PLL参数"""
        param_ranges = [
            (100, 1000),  # wn
            (0.6, 1.2),  # zeta
            (1.0, 5.0),  # Kd
            (100, 1000),  # Kvco
            (1e-5, 1e-3),  # tau1
            (1e-6, 1e-4)  # tau2
        ]

        params = []
        for i, (low, high) in enumerate(param_ranges):
            # 将[-2, 2]映射到[0, 1]
            norm_val = (normalized_params[i] + 2) / 4
            norm_val = np.clip(norm_val, 0, 1)

            if i in [4, 5]:  # tau1, tau2使用对数映射
                log_low, log_high = np.log10(low), np.log10(high)
                param_val = 10 ** (log_low + norm_val * (log_high - log_low))
            else:
                param_val = low + norm_val * (high - low)

            params.append(param_val)

        return np.array(params)

    def calculate_diversity(self):
        """计算粒子群多样性"""
        if self.n_particles <= 1:
            return 0
        center = np.mean(self.positions, axis=0)
        diversity = np.mean([np.linalg.norm(pos - center) for pos in self.positions])
        return diversity

    def get_particle_info(self, iteration):
        """获取粒子详细信息"""
        info = {
            'iteration': iteration,
            'positions': self.positions.copy(),
            'velocities': self.velocities.copy(),
            'personal_best_scores': self.personal_best_scores.copy(),
            'global_best_score': self.global_best_score,
            'global_best_position': self.global_best_position.copy() if self.global_best_position is not None else None,
            'diversity': self.calculate_diversity(),
            'inertia_weight': self.w,
            'best_particle_idx': np.argmax(self.personal_best_scores),
            'worst_particle_idx': np.argmin(self.personal_best_scores),
            'avg_personal_best': np.mean(self.personal_best_scores),
            'std_personal_best': np.std(self.personal_best_scores)
        }
        return info

    def print_detailed_iteration(self, iteration, show_particles=5):
        """打印详细的迭代信息"""
        print(f"\n  详细迭代 {iteration}:")
        print(f"   全局最优得分: {self.global_best_score:.4f}")
        print(f"   粒子群多样性: {self.calculate_diversity():.4f}")
        print(f"   惯性权重: {self.w:.4f}")

        # 显示个体最优得分统计
        print(f"   个体最优得分: 平均={np.mean(self.personal_best_scores):.2f}, "
              f"标准差={np.std(self.personal_best_scores):.2f}")
        print(f"   得分范围: [{np.min(self.personal_best_scores):.2f}, {np.max(self.personal_best_scores):.2f}]")

        # 显示前几个最好的粒子
        best_indices = np.argsort(self.personal_best_scores)[-show_particles:][::-1]
        print(f"   前{show_particles}个最佳粒子:")

        param_names = ['ωₙ', 'ζ', 'Kd', 'Kvco', 'τ₁', 'τ₂']
        for i, idx in enumerate(best_indices):
            actual_params = self.normalize_params(self.personal_best_positions[idx])
            print(f"     粒子{idx:2d}: 得分={self.personal_best_scores[idx]:6.2f}, 参数=[", end="")
            for j, (name, val) in enumerate(zip(param_names, actual_params)):
                if j < len(param_names) - 1:
                    print(f"{val:.2e}, ", end="")
                else:
                    print(f"{val:.2e}]")

        # 显示速度统计
        vel_norms = [np.linalg.norm(vel) for vel in self.velocities]
        print(f"   速度统计: 平均={np.mean(vel_norms):.4f}, 最大={np.max(vel_norms):.4f}")

    def optimize(self, objective_func, max_iterations=200, tolerance=1e-6, verbose=True, detailed_first_n=20):
        """执行优化 - 增加详细输出选项"""

        if verbose:
            print("  开始自适应粒子群优化...")
            print(f"粒子数: {self.n_particles}, 最大迭代: {max_iterations}")
            print(f"详细显示前 {detailed_first_n} 轮迭代")
            print("-" * 80)

        stagnation_count = 0
        last_best_score = -float('inf')

        for iteration in range(max_iterations):
            # 评估所有粒子
            for i in range(self.n_particles):
                # 将归一化参数转换为实际参数
                actual_params = self.normalize_params(self.positions[i])
                score = objective_func(actual_params)

                # 更新个体最优
                if score > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i].copy()

                # 更新全局最优
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()

            # 记录详细信息
            if iteration < detailed_first_n:
                iter_info = self.get_particle_info(iteration)
                self.detailed_history.append(iter_info)

                # 打印详细信息
                if verbose:
                    self.print_detailed_iteration(iteration)

            # 自适应调整惯性权重
            diversity = self.calculate_diversity()
            self.w = self.w_min + (self.w_max - self.w_min) * diversity / 2.0

            # 更新速度和位置
            for i in range(self.n_particles):
                # 随机因子
                r1, r2 = np.random.random(2)

                # 速度更新
                self.velocities[i] = (
                        self.w * self.velocities[i] +
                        self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) +
                        self.c2 * r2 * (self.global_best_position - self.positions[i])
                )

                # 限制速度
                max_vel = 0.2
                self.velocities[i] = np.clip(self.velocities[i], -max_vel, max_vel)

                # 位置更新
                self.positions[i] += self.velocities[i]

                # 边界处理
                for j in range(self.n_dimensions):
                    if self.positions[i][j] < self.bounds[j][0]:
                        self.positions[i][j] = self.bounds[j][0]
                        self.velocities[i][j] = 0
                    elif self.positions[i][j] > self.bounds[j][1]:
                        self.positions[i][j] = self.bounds[j][1]
                        self.velocities[i][j] = 0

            # 记录历史
            self.score_history.append(self.global_best_score)
            self.diversity_history.append(diversity)

            # 停滞检测
            if abs(self.global_best_score - last_best_score) < tolerance:
                stagnation_count += 1
            else:
                stagnation_count = 0

            last_best_score = self.global_best_score

            # 简化输出（超过详细显示范围后）
            if verbose and iteration >= detailed_first_n and iteration % 20 == 0:
                print(f"Iter {iteration:3d}: 最佳得分={self.global_best_score:6.2f}, "
                      f"多样性={diversity:.4f}, 惯性权重={self.w:.3f}")

            # 早停条件
            if stagnation_count > 30:
                if verbose:
                    print(f"\n在第{iteration}轮收敛，停滞{stagnation_count}轮")
                break

            # 重启机制
            if iteration > 0 and iteration % 100 == 0 and diversity < 0.1:
                if verbose:
                    print(f"\n第{iteration}轮执行粒子群重启...")
                # 保留最优粒子，重启其他粒子
                best_particle_idx = np.argmax(self.personal_best_scores)
                self.positions = np.random.uniform(
                    [b[0] for b in self.bounds],
                    [b[1] for b in self.bounds],
                    (self.n_particles, self.n_dimensions)
                )
                self.positions[0] = self.personal_best_positions[best_particle_idx]

                self.velocities = np.random.uniform(
                    -0.1, 0.1, (self.n_particles, self.n_dimensions)
                )

        if verbose:
            print(f"\n  优化完成! 最终得分: {self.global_best_score:.2f}")

        return self.normalize_params(self.global_best_position), self.global_best_score


class AdaptivePLLOptimizer:
    """自适应PLL参数优化器"""

    def __init__(self):
        self.env = PLLEnvironment()
        self.optimizer = AdaptivePSOOptimizer()

        self.optimization_history = []
        self.best_params_per_condition = []

    def optimize_for_condition(self, delta_f, delta_theta, noise_level=0.005, max_iter=200, detailed_first_n=20):
        """为特定初始条件优化参数"""

        # 设置环境条件
        self.env.reset(delta_f, delta_theta, noise_level)

        # 定义目标函数
        def objective(params):
            return self.env.evaluate_performance(params)

        # 重置优化器
        self.optimizer.reset()

        # 执行优化
        best_params, best_score = self.optimizer.optimize(objective, max_iter, verbose=False,
                                                          detailed_first_n=detailed_first_n)

        # 记录结果
        result = {
            'conditions': (delta_f, delta_theta, noise_level),
            'best_params': best_params,
            'best_score': best_score,
            'score_history': self.optimizer.score_history.copy(),
            'detailed_history': self.optimizer.detailed_history.copy()  # 保存详细历史
        }

        self.optimization_history.append(result)
        return best_params, best_score

    def comprehensive_optimization(self, n_conditions=cf.n_conditions, detailed_first_n=20):
        """综合优化：测试多种初始条件"""

        print("  开始综合PLL参数优化...")
        print(f"测试条件数: {n_conditions}")
        print(f"每个条件详细显示前 {detailed_first_n} 轮迭代")
        print("=" * 80)

        results = []

        test_conditions = []
        if cf.use_random_condition:
            # 生成测试条件
            for i in range(n_conditions):
                delta_f = np.random.uniform(cf.delta_f_min, cf.delta_f_max)
                delta_theta = np.random.uniform(cf.delta_theta_min, cf.delta_theta_max)
                noise_level = np.random.uniform(cf.noise_level_min, cf.noise_level_max)
                test_conditions.append((delta_f, delta_theta, noise_level))
        else:
            test_conditions = cf.test_conditions

        # 为每个条件优化参数
        for i, (df, dt, noise) in enumerate(test_conditions):
            print(f"\n{'=' * 20} 条件 {i + 1}/{n_conditions} {'=' * 20}")
            print(f"初始条件: Δf={df:.1f}Hz, Δθ={dt:.3f}rad, 噪声={noise:.3f}")

            best_params, best_score = self.optimize_for_condition(df, dt, noise, detailed_first_n=detailed_first_n)

            # 详细评估性能
            score, details = self.env.evaluate_performance(best_params, return_details=True)

            result = {
                'condition_id': i,
                'delta_f': df,
                'delta_theta': dt,
                'noise_level': noise,
                'best_params': best_params,
                'best_score': score,
                'details': details
            }

            results.append(result)

            print(f"\n  条件 {i + 1} 最终结果:")
            print(f"   最优得分: {score:.2f}")
            print(f"   锁定成功: {'是' if details['lock_success'] else '否'}")
            if details['lock_success']:
                print(f"   锁定时间: {details['lock_time'] * 1000:.2f}ms")
                print(f"   稳态误差: {details['steady_state_error']:.4f}rad")
                print(f"   最大过冲: {details['overshoot']:.5f} rad")
                print(f"   相位振荡性: {details['oscillation']:.5f} rad")

            # 显示最优参数
            param_names = ['ωₙ (rad/s)', 'ζ', 'Kd', 'Kvco', 'τ₁ (s)', 'τ₂ (s)']
            print(f"   最优参数:")
            for name, val in zip(param_names, best_params):
                print(f"     {name:12}: {val:.3e}")

        self.best_params_per_condition = results
        print(f"\n  综合优化完成!")

        return results

    def analyze_parameter_robustness(self):
        """分析参数鲁棒性"""
        if not self.best_params_per_condition:
            print("请先运行comprehensive_optimization()")
            return

        # 提取所有最优参数
        all_params = np.array([r['best_params'] for r in self.best_params_per_condition])
        param_names = ['ωₙ (rad/s)', 'ζ', 'Kd', 'Kvco', 'τ₁ (s)', 'τ₂ (s)']

        print("\n  参数鲁棒性分析:")
        print("=" * 60)

        for i, name in enumerate(param_names):
            values = all_params[:, i]
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            cv = std_val / mean_val if mean_val != 0 else float('inf')

            print(f"{name:12s}: 均值={mean_val:.2e}, 标准差={std_val:.2e}")
            print(f"             范围=[{min_val:.2e}, {max_val:.2e}], CV={cv:.3f}")

        # 成功率统计
        success_count = sum(1 for r in self.best_params_per_condition if r['details']['lock_success'])
        success_rate = success_count / len(self.best_params_per_condition)

        avg_score = np.mean([r['best_score'] for r in self.best_params_per_condition])

        print(f"\n   整体性能:")
        print(f"   成功率: {success_rate:.1%} ({success_count}/{len(self.best_params_per_condition)})")
        print(f"   平均得分: {avg_score:.2f}")

        return all_params, param_names


# 主运行函数
def main():
    """主函数：演示改进的PLL优化系统"""

    # 创建优化器
    optimizer = AdaptivePLLOptimizer()

    # 执行综合优化 - 只测试3个条件以便观察详细输出
    results = optimizer.comprehensive_optimization(n_conditions=3, detailed_first_n=20)

    # 分析参数鲁棒性
    all_params, param_names = optimizer.analyze_parameter_robustness()

    return optimizer, results


if __name__ == "__main__":
    # 运行主程序
    optimizer, results = main()
