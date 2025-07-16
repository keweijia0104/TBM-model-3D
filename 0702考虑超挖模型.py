import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ShieldInteractionModel:
    """
    【已修改】增加了 over_excavation_mode 开关，并恢复了 element_data 的输出
    """
    def __init__(self, params, over_excavation_mode=False):
        # 基础参数
        self.R = params['R']
        self.Lf = params['Lf']
        self.Lr = params['Lr']
        self.LG = params['LG']
        self.Rp = params['Rp']
        self.G = params['G'] * 9.81e3
        self.xi = params['xi']
        self.mu = params['mu']
        self.gamma_soil = params['gamma_soil'] * 1e3
        self.H = params['H']
        self.K0 = params['K0']
        self.Kint = params['Kint']
        self.Kmin = params['Kmin']
        self.Kmax = params['Kmax']
        self.alpha_r_initial = np.deg2rad(params['alpha_r0'])
        self.beta_r_initial = np.deg2rad(params['beta_r0'])
        
        # 【新增】超挖模式开关
        self.over_excavation_mode = over_excavation_mode

        # 离散化参数
        self.n_theta, self.n_v, self.n_rho = 48, 48, 24
        self.L_total = self.Lf + self.Lr
        self.A_shell_element = (2 * np.pi * self.R * self.L_total) / (self.n_theta * self.n_v)
        self.theta_shell = np.linspace(0, 2 * np.pi, self.n_theta, endpoint=False)
        self.v_shell = np.linspace(-self.Lr, self.Lf, self.n_v)
        self.U_total = np.zeros((self.n_v, self.n_theta))
        self.W_total = np.zeros((self.n_v, self.n_theta))

    def _get_ground_reaction_coeff(self, delta: float) -> float:
        delta_p = 0.01807
        Kint, Kmin, Kmax = self.Kint, self.Kmin, self.Kmax
        if Kmax - Kint == 0: lam = float('inf')
        else: lam = (Kint - Kmin) / (Kmax - Kint)
        delta_a = -lam * delta_p
        
        if delta <= delta_a: K = Kmin
        elif delta < 0.0:
            K = Kint if delta_a == 0 else Kmin + (delta - delta_a) * (Kint - Kmin) / (-delta_a)
        elif delta < delta_p:
            K = Kint if delta_p == 0 else Kint + delta * (Kmax - Kint) / delta_p
        else: K = Kmax
        return K

    def _calculate_earth_pressure(self, U: np.ndarray, W: np.ndarray):
        F2, M2 = np.zeros(3), np.zeros(3)
        element_data = [] 
        for m, v_m in enumerate(self.v_shell):
             for i, theta in enumerate(self.theta_shell):
                sigma_w0, sigma_u0 = self._initial_soil_pressures(theta)
                delta_u, delta_w = U[m, i], W[m, i]
                
                Ku_by_disp = self._get_ground_reaction_coeff(-delta_u) if (np.pi / 2) <= theta <= 3/2*np.pi else self._get_ground_reaction_coeff(delta_u)
                
                # 【修改点】如果开启超挖模式，则强制特定区域为主动土压力
                # 角度范围: 135度 to 225度
                theta_min_rad, theta_max_rad = np.deg2rad(135), np.deg2rad(225)
                # 纵向范围: 2.5m to 3.5m
                v_min, v_max = 2.5, 3.5

                if self.over_excavation_mode and (v_min <= v_m <= v_max) and (theta_min_rad <= theta <= theta_max_rad):
                    Ku = self.Kmin
                else:
                    Ku = Ku_by_disp

                Kw = self._get_ground_reaction_coeff(-delta_w) if np.pi <= theta <= (2 * np.pi) else self._get_ground_reaction_coeff(delta_w)
                
                dF = np.array([Ku * sigma_u0, 0.0, Kw * sigma_w0]) * self.A_shell_element
                r_vec = np.array([self.R * np.cos(theta), v_m, -self.R * np.sin(theta)])
                F2 += dF
                M2 += np.cross(r_vec, dF)

                element_data.append({'m': m, 'i': i, 'v': v_m, 'theta_deg': np.rad2deg(theta), 'Ku': Ku, 'Kw': Kw, 'Force_u_N': dF[0]})
        return F2, M2, element_data
    
    def solve_for_thrust_differences(self, target_alpha_r_deg: float, target_beta_r_deg: float, d_alpha_h_deg: float, d_beta_h_deg: float):
        alpha_r, beta_r = np.deg2rad(target_alpha_r_deg), np.deg2rad(target_beta_r_deg)
        d_alpha_h, d_beta_h = np.deg2rad(d_alpha_h_deg), np.deg2rad(d_beta_h_deg)
        d_alpha, d_beta = alpha_r - self.alpha_r_initial, beta_r - self.beta_r_initial
        alpha_f, beta_f = alpha_r + d_alpha_h, beta_r + d_beta_h

        def residual_disp(vars_disp):
            du_s, dw_s = vars_disp
            du, dw = np.empty_like(self.U_total), np.empty_like(self.W_total)
            for m, v_m in enumerate(self.v_shell):
                # ***** MODIFICATION START 1 *****
                # 根据图片更新位移计算公式
                if v_m >= 0:
                    dw[m, :] = dw_s + v_m * np.tan(d_beta_h + d_beta)
                    du[m, :] = du_s + v_m * np.tan(d_alpha_h + d_alpha)
                else:
                    dw[m, :], du[m, :] = dw_s + v_m * np.tan(d_beta), du_s + v_m * np.tan(d_alpha)
                # ***** MODIFICATION END 1 *****
            F1, _ = self._calculate_self_weight_21(beta_r, d_alpha_h, d_beta_h)
            F3, _ = self._calculate_cutter_pressure(alpha_f, beta_f, d_alpha_h, d_beta_h)
            F2, _, _ = self._calculate_earth_pressure(self.U_total + du, self.W_total + dw)
            return [F1[0] + F2[0] + F3[0], F1[2] + F2[2] + F3[2]]
        
        sol_disp = root(residual_disp, [0, 0], method='hybr')
        if not sol_disp.success: return False, None

        du_s_sol, dw_s_sol = sol_disp.x
        du_final, dw_final = np.empty_like(self.U_total), np.empty_like(self.W_total)
        for m, v_m in enumerate(self.v_shell):
            # ***** MODIFICATION START 2 *****
            # 同样根据图片更新最终位移的计算
            if v_m >= 0:
                dw_final[m, :] = dw_s_sol + v_m * np.tan(d_beta_h + d_beta)
                du_final[m, :] = du_s_sol + v_m * np.tan(d_alpha_h + d_alpha)
            else:
                dw_final[m, :], du_final[m, :] = dw_s_sol + v_m * np.tan(d_beta), du_s_sol + v_m * np.tan(d_alpha)
            # ***** MODIFICATION END 2 *****

        F1_f, M1_f = self._calculate_self_weight_21(beta_r, d_alpha_h, d_beta_h)
        F3_f, M3_f = self._calculate_cutter_pressure(alpha_f, beta_f, d_alpha_h, d_beta_h)
        F2_f, M2_f, element_data = self._calculate_earth_pressure(self.U_total + du_final, self.W_total + dw_final)
        
        Mu_req, Mw_req = -(M1_f[0] + M2_f[0] + M3_f[0]), -(M1_f[2] + M2_f[2] + M3_f[2])
        s_sin, s_cos, kN_to_N = 4.26197, 2.84776, 1e3
        diff_T_B = Mu_req / (self.Rp * s_cos * kN_to_N)
        diff_R_L = Mw_req / (self.Rp * s_sin * kN_to_N)
        return True, {'diff_T_B': diff_T_B, 'diff_R_L': diff_R_L, 'element_data': element_data}

    def _calculate_self_weight_21(self, beta_r: float, alpha_h: float, beta_h: float):
        F1 = np.array([0.0, self.G * np.sin(beta_r), self.G * np.cos(beta_r)])
        l1 = np.array([-self.LG * np.sin(alpha_h) * np.cos(beta_h), -self.LG * np.cos(alpha_h) * np.cos(beta_h), -self.LG * np.cos(alpha_h) * np.sin(beta_h)])
        return F1, np.cross(F1, l1)

    def _calculate_cutter_pressure(self, alpha_f: float, beta_f: float, alpha_h: float, beta_h: float):
        drho, dtheta = self.R / self.n_rho, 2.0 * np.pi / self.n_theta
        F3_total, M3_total = np.zeros(3), np.zeros(3)
        for i in range(1, self.n_theta + 1):
            theta = (2 * i - 1) * np.pi / self.n_theta
            ct, st = np.cos(theta), np.sin(theta)
            for j in range(1, self.n_rho + 1):
                rho = (2 * j - 1) * self.R / (2 * self.n_rho)
                dA = rho * drho * dtheta
                sigma = self.gamma_soil * self.K0 * (self.H + self.R - rho * ct)
                force_direction_vec = np.array([-2 * (1 - self.xi) * self.mu * ct, -1.0, 2 * (1 - self.xi) * self.mu * st])
                force_scale = sigma * np.cos(beta_f) * np.cos(alpha_h) * np.cos(beta_h) * dA
                dF = force_scale * force_direction_vec
                r_vec = np.array([-rho * st, -self.Lf, -rho * ct])
                F3_total += dF
                M3_total += np.cross(dF, r_vec)
        return F3_total, M3_total

    def _initial_soil_pressures(self, theta: float) -> tuple[float, float]:
        alpha = abs(((np.pi / 2 - theta) + np.pi) % (2 * np.pi) - np.pi)
        sigma_w_mag = self.gamma_soil * self.H + (4.0 * self.G) / (np.pi**2 * (2*self.R) * self.L_total) * alpha
        sigma_u_mag = self.K0 * sigma_w_mag
        return 0.33 * np.sin(theta) * sigma_w_mag, 0.33 * -np.cos(theta) * sigma_u_mag

def plot_shield_heatmap(df, kint, case_name, output_folder):
    """
    【新增】绘制并保存盾壳展开的热力图
    """
    df_sorted = df.sort_values(by=['m', 'i'])
    v_ticks = np.unique(df_sorted['v'])
    theta_ticks_deg = np.unique(df_sorted['theta_deg'])
    
    data_sets = {}
    left_indices = (theta_ticks_deg >= 90) & (theta_ticks_deg <= 270)
    data_sets['左侧'] = {'indices': left_indices, 'theta': theta_ticks_deg[left_indices]}
    
    right_indices_mask = ~left_indices
    right_theta = theta_ticks_deg[right_indices_mask]
    right_indices_sorted = np.concatenate((np.where(right_theta > 270)[0], np.where(right_theta < 90)[0]))
    data_sets['右侧'] = {'indices': right_indices_mask, 'theta': right_theta[right_indices_sorted], 'sorted_idx': right_indices_sorted}
    
    max_abs_force = df['Force_u_N'].abs().max() or 1.0

    for side, data in data_sets.items():
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        
        ku_grid = df_sorted.pivot(index='m', columns='i', values='Ku')
        force_u_grid = df_sorted.pivot(index='m', columns='i', values='Force_u_N')

        if side == '右侧':
            ku_side = ku_grid.iloc[:, data['indices']].iloc[:, data['sorted_idx']]
            force_side = force_u_grid.iloc[:, data['indices']].iloc[:, data['sorted_idx']]
        else:
            ku_side = ku_grid.iloc[:, data['indices']]
            force_side = force_u_grid.iloc[:, data['indices']]

        force_red = force_side.where(ku_side > kint)
        force_blue = force_side.where(ku_side < kint)

        ax.imshow(force_red.abs(), cmap='Reds', vmin=0, vmax=max_abs_force, aspect='auto')
        ax.imshow(force_blue.abs(), cmap='Blues', vmin=0, vmax=max_abs_force, aspect='auto')

        ax.set_xticks(np.arange(len(data['theta'])))
        ax.set_yticks(np.arange(len(v_ticks)))
        ax.set_xticklabels([f'{t:.0f}°' for t in data['theta']], rotation=90, fontsize=8)
        ax.set_yticklabels([f'{v:.2f}' for v in v_ticks], fontsize=8)
        ax.set(xlabel='周向角度 θ (°)', ylabel='纵向坐标 v (m)', title=f'盾壳{side}水平土压力分布 Force_u (N)\n工况: {case_name}')
        
        for i in range(len(v_ticks)):
            for j in range(len(data['theta'])):
                ax.text(j, i, f'{force_side.iloc[i, j]:.0f}', ha='center', va='center', color='black', fontsize=6)
        
        fig.tight_layout()
        filename = os.path.join(output_folder, f"Heatmap_{side}_{case_name}.svg")
        plt.savefig(filename, format='svg')
        plt.close(fig)

def run_analysis(base_params, d_alpha_h_range, targets, over_excavation_mode, output_folder):
    mode_str = "考虑内侧超挖" if over_excavation_mode else "不考虑超挖"
    print(f"\n===== 开始计算工况: {mode_str} =====")
    
    diff_T_B_list, diff_R_L_list = [], []
    
    # ***** MODIFICATION START 3 *****
    # 核心逻辑修改：迭代铰接角d_alpha_h, 保持前盾姿态角alpha_f不变, 反算后盾姿态角alpha_r
    for i, d_alpha_h in enumerate(d_alpha_h_range):
        # alpha_f = alpha_r + d_alpha_h  =>  alpha_r = alpha_f - d_alpha_h
        current_alpha_r = targets['alpha_f'] - d_alpha_h
        # beta_f = beta_r + d_beta_h, 这里 d_beta_h 为 0.0
        current_beta_r = targets['beta_f'] - 0.0
        
        model = ShieldInteractionModel(base_params, over_excavation_mode=over_excavation_mode)
        success, results = model.solve_for_thrust_differences(
            current_alpha_r, current_beta_r, d_alpha_h, 0.0)
        
        if success:
            diff_T_B_list.append(results['diff_T_B'])
            diff_R_L_list.append(results['diff_R_L'])
            
            # ***** MODIFICATION START 4 *****
            # 为每个计算步都生成热力图 (移除 if 条件)
            case_name = f"{mode_str}_d_alpha_h_{d_alpha_h:.2f}"
            print(f"  -> 为铰接角 {d_alpha_h:.2f}° (对应后盾姿态角 {current_alpha_r:.2f}°) 生成详细盾壳图...")
            df_elements = pd.DataFrame(results['element_data'])
            plot_shield_heatmap(df_elements, model.Kint, case_name, output_folder)
            # ***** MODIFICATION END 4 *****
        else:
            print(f"  -> {mode_str}工况下 d_alpha_h = {d_alpha_h:.2f}° 求解失败。")
            diff_T_B_list.append(np.nan)
            diff_R_L_list.append(np.nan)
    # ***** MODIFICATION END 3 *****

    s_T_B = pd.Series(diff_T_B_list).interpolate(method='linear', limit_direction='both')
    s_R_L = pd.Series(diff_R_L_list).interpolate(method='linear', limit_direction='both')
    
    print(f"===== 工况计算完成: {mode_str} =====")
    return s_T_B.to_list(), s_R_L.to_list()


def main():
    base_params = {
        'R': 2.455, 'Lf': 3.5, 'Lr': 4, 'LG': 2, 'Rp': 2.14, 'G': 180,
        'xi': 0.32, 'mu': 0.30, 'gamma_soil': 19, 'H': 32, 'K0': 0.35,
        'Kint': 0.666, 'Kmin': 0.499, 'Kmax': 2.002,
        'alpha_r0': 0, 'beta_r0': 0,
    }

    # ***** MODIFICATION START 5 *****
    # 目标参数从后盾姿态角(alpha_r)变为前盾姿态角(alpha_f)
    targets = {'alpha_f': -1.5, 'beta_f': 0.0}
    
    # 将计算范围改为-3到0，间隔0.1
    d_alpha_h_range = np.arange(-3.0, 0.0 + 0.1, 0.1)
    
    # 为新逻辑创建新的输出文件夹
    comparison_plot_folder = 'SVG_Comparison_Plots_New_Logic'
    # 将盾壳图文件夹重命名
    heatmap_folder = '考虑超挖前盾固定盾壳图01'
    if not os.path.exists(comparison_plot_folder): os.makedirs(comparison_plot_folder)
    if not os.path.exists(heatmap_folder): os.makedirs(heatmap_folder)
    # ***** MODIFICATION END 5 *****

    # 执行两种工况的计算
    tb_normal, rl_normal = run_analysis(base_params, d_alpha_h_range, targets, over_excavation_mode=False, output_folder=heatmap_folder)
    tb_over_exc, rl_over_exc = run_analysis(base_params, d_alpha_h_range, targets, over_excavation_mode=True, output_folder=heatmap_folder)
    
    # 绘制并保存对比图
    print("\n===== 正在生成最终对比图... =====")
    plot_x_data = -d_alpha_h_range

    fig1, ax1 = plt.subplots(figsize=(12, 8), dpi=150)
    ax1.plot(plot_x_data, tb_normal, 'o-', label='不考虑超挖', color='royalblue', linewidth=2.5)
    ax1.plot(plot_x_data, tb_over_exc, 's--', label='考虑内侧超挖', color='deepskyblue', linewidth=2.5)
    ax1.set(xlabel='水平铰接角 $-d_{αh}$ (°)', ylabel='所需顶-底推力差值 (T-B) (kN)', title='铰接角与顶底推力差的关系对比 (新逻辑)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(comparison_plot_folder, "Comparison_T-B_New_Logic.svg"), format='svg', bbox_inches='tight')
    
    fig2, ax2 = plt.subplots(figsize=(12, 8), dpi=150)
    # ***** MODIFICATION START 6 *****
    # 左右推力差乘以负号 (移除原有的负号)
    ax2.plot(plot_x_data, np.array(rl_normal), 'o-', label='不考虑超挖', color='crimson', linewidth=2.5)
    ax2.plot(plot_x_data, np.array(rl_over_exc), 's--', label='考虑内侧超挖', color='salmon', linewidth=2.5)
    # ***** MODIFICATION END 6 *****
    ax2.set(xlabel='水平铰接角 $-d_{αh}$ (°)', ylabel='所需左-右推力差值 (L-R) (kN)', title='铰接角与左右推力差的关系对比 (新逻辑)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(comparison_pl_folder, "Comparison_L-R_New_Logic.svg"), format='svg', bbox_inches='tight')

    print(f"所有对比图已保存至 '{comparison_plot_folder}' 文件夹。")
    print(f"所有盾壳热力图已保存至 '{heatmap_folder}' 文件夹。")
    print("="*40 + "\n所有任务执行完毕。")
    plt.show()

if __name__ == '__main__':
    main()