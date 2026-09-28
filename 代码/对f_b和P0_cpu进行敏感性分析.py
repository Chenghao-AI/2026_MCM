import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =========================
# 1. 核心数学模型
# =========================
def get_signals_noisy(t):
    rsrp_base = -95 + 15 * np.sin(2 * np.pi * t / 3600)
    snr_base = 10 + 8 * np.cos(2 * np.pi * t / 1800)
    seed_val = int(abs(t * 100)) % 10**8
    rng = np.random.default_rng(seed_val)
    shadowing = 3.0 * np.sin(2 * np.pi * t / 600) 
    rsrp = np.clip(rsrp_base + shadowing + rng.normal(0, 1.5), -120, -60)
    snr = np.clip(snr_base + shadowing*0.5 + rng.normal(0, 2.0), -2, 28)
    return float(rsrp), float(snr)

def calculate_all_powers(t, T, p):
    # --- CPU Power ---
    u = 1.0 if T <= p["T0_cpu"] else (1 - p["u_min"]) * np.exp(-(T - p["T0_cpu"]) / p["tau"]) + p["u_min"]
    p_cpu = (p["P0_cpu"] + (p["k0"] + p["alpha_cpu"] * p["I_down"]) * (p["f0"] * u / 1e9))/p["eta_0"]
    
    # --- Network Power (p_b) ---
    # 核心公式: p_b = (gamma * I_down + f_b * PE_0) / eta_0
    # 注意：此处 PE_0 取值为 0.15 (参考原代码中的硬编码值)
    p_b = (p["gamma"] * p["I_down"] + p["f_b"] * 0.15) / p["eta_0"]
    p_RF = p_b
    p_down = 0.0
    
    # --- Display Power ---
    p_dis = 0.0
    
    return p_RF, p_down, p_cpu, p_dis

# =========================
# 2. ODE 定义
# =========================
def battery_ode(t, y, p):
    SoC, T = y
    p_RF, p_down, p_cpu, p_dis = calculate_all_powers(t, T, p)
    P_tot = p_RF + p_down + p_cpu + p_dis
    E_joules = p["E_base_J"] * (0.0026*T**3 - 1.2176*T**2 + 73.8623*T + 9933.3511) / 9286.8384 # 简化系数
    
    dSoC_dt = -P_tot / E_joules
    dT_dt = ((1-p["eta_0"])*p_cpu+(1-p["eta_dis"])*p_dis + (1-p["eta_net"])*(p_RF+p_down) - p["lambda"]*(T - p["T_env"])) / p["C_eff"]
    return [dSoC_dt, dT_dt]

def soc_depleted_event(t, y, p):
    return y[0]
soc_depleted_event.terminal = True
soc_depleted_event.direction = -1

# =========================
# 3. 敏感性分析
# =========================

base_params = {
    # 固定参数
    "gamma": 2.0000, "P0_cpu": 0.0500, "alpha_cpu": 0.0100, "k1": 0.1025, "k2": 0.0106,
    # Screen Off 特定
    "k0": 0.16, "I_down": 0.01,
    # 系统常量
    "eta_0": 0.35, "eta_net": 0.4, "eta_dis": 0.8, "C_eff": 120.0, "T_env": 25.0, "lambda": 0.3,
    "E_base_J": 87240, 
    "f0": 2.0e9, "u_min": 0.6, "T0_cpu": 45.0, "tau": 5.0,
    "f_b": 0.02948 # 设定 base_param 中 f_b 为 0.02948
}

def get_time_to_empty(params):
    # 仿真足够长的时间以确保耗尽，例如 500 小时
    sol = solve_ivp(
        battery_ode, [0, 500*3600], [1.0, 25.0], 
        args=(params,), events=soc_depleted_event, max_step=300
    )
    if sol.t_events and sol.t_events[0].size > 0:
        return sol.t_events[0][0] / 3600.0
    else:
        # 如果未耗尽，返回结束时间（应避免这种情况）
        return sol.t[-1] / 3600.0

def run_sensitivity_analysis():
    # 实验 1: 固定 f_b, 变化 P0_cpu (-100% ~ +20%, step 5%)
    print("Running Experiment 1: P0_cpu Sensitivity...")
    fb_fixed = 0.02948
    p0_cpu_base = 0.0500
    
    p0_cpu_values = []
    tte_values_1 = []
    
    # range(-20, 5) -> -20, -19, ..., 0, ..., 4 (共 25 个点)
    # 若需凑够 29 个点，可能需要调整范围，但此处严格遵循 "-100% ~ +20%"
    for i in range(-20, 5): 
        multiplier = i * 0.05
        val = p0_cpu_base * (1 + multiplier)
        if val < 0: val = 0 
        
        p = base_params.copy()
        p["f_b"] = fb_fixed
        p["P0_cpu"] = val
        
        tte = get_time_to_empty(p)
        p0_cpu_values.append(val)
        tte_values_1.append(tte)
        print(f"  P0_cpu={val:.6f} ({multiplier*100:+.0f}%) -> TTE={tte:.4f} h")

    # 实验 2: 固定 P0_cpu, 变化 f_b (-100% ~ +20%, step 5%)
    print("\nRunning Experiment 2: f_b Sensitivity...")
    p0_cpu_fixed = 0.0500
    fb_base = 0.02948
    
    fb_values = []
    tte_values_2 = []
    
    for i in range(-20, 5): 
        multiplier = i * 0.05
        val = fb_base * (1 + multiplier)
        if val < 0: val = 0
        
        p = base_params.copy()
        p["P0_cpu"] = p0_cpu_fixed
        p["f_b"] = val
        
        tte = get_time_to_empty(p)
        fb_values.append(val)
        tte_values_2.append(tte)
        print(f"  f_b={val:.6f} ({multiplier*100:+.0f}%) -> TTE={tte:.4f} h")

    # 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1
    axes[0].plot(p0_cpu_values, tte_values_1, color='blue', marker='o', linestyle='-', linewidth=2, markersize=8)
    axes[0].set_title("Impact of Background Tasks: P0_cpu")
    axes[0].set_xlabel("P0_cpu (Actual Value)")
    axes[0].set_ylabel("Time to Empty (hour)")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2
    axes[1].plot(fb_values, tte_values_2, color='blue', marker='o', linestyle='-', linewidth=2, markersize=8)
    axes[1].set_title("Impact of Background Tasks: f_b")
    axes[1].set_xlabel("f_b (Actual Value)")
    axes[1].set_ylabel("Time to Empty (hour)")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_sensitivity_analysis()
