import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =========================
# 1. 核心数学模型与噪声信号
# =========================
def get_signals_noisy(t):
    rsrp_base = -95 + 15 * np.sin(2 * np.pi * t / 3600)
    snr_base = 10 + 8 * np.cos(2 * np.pi * t / 1800)
    
    # 确定性伪随机噪声，确保求解稳定性
    seed_val = int(abs(t * 100)) % 10**8
    rng = np.random.default_rng(seed_val)
    
    shadowing = 3.0 * np.sin(2 * np.pi * t / 600) 
    rsrp = np.clip(rsrp_base + shadowing + rng.normal(0, 1.5), -120, -60)
    snr = np.clip(snr_base + shadowing*0.5 + rng.normal(0, 2.0), -2, 28)
    return float(rsrp), float(snr)

def calculate_all_powers(t, T, p):
    rsrp, snr = get_signals_noisy(t)
    # T0_cpu 为温控保护阈值 (45°C), T 为当前温度
    u = 1.0 if T <= p["T0_cpu"] else (1 - p["u_min"]) * np.exp(-(T - p["T0_cpu"]) / p["tau"]) + p["u_min"]
    
    # Network
    PL = p["P1_dl_dbm"] - rsrp
    Pup_W = 10**((np.minimum(p["P0_dbm"] + PL, p["Pmax_dbm"]) - 30)/10)
    eta = 0.4
    v_down = p["rho"] * p["B_hz"] * np.log2(1 + 10**(snr/10))
    p_down = ((p["gamma"] * p["I_down"]) )/p["eta_net"]
    p_RF=((Pup_W / eta) / (v_down / 1e6) * p["I_down"])
    
    # CPU & Display
    p_cpu = (p["P0_cpu"] + (p["k0"] + p["alpha_cpu"] * p["I_down"]) * (p["f0"] * u / 1e9))/p["eta_0"]
    p_dis =((p["k1"] * p["L"] + p["k2"] * p["R0"] * u) * p["S"] * p["p"])/p["eta_dis"]
    
    return p_RF,p_down, p_cpu, p_dis

# =========================
# 2. ODE 定义与事件检测
# =========================
def battery_ode(t, y, p):
    SoC, T = y
    p_RF,p_down, p_cpu, p_dis = calculate_all_powers(t, T, p)
    P_tot = p_RF+p_down + p_cpu + p_dis
    
    # 电池容量修正
    E_joules = p["E_base_J"] * (0.0026*T**3 - 1.2176*T**2 + 73.8623*T + 9933.3511) / 9286.8384 # 简化系数
    
    dSoC_dt = -P_tot / E_joules
    dT_dt = ((1-p["eta_0"])*p_cpu+(1-p["eta_dis"])*p_dis + (1-p["eta_net"])*(p_RF+p_down) - p["lambda"]*(T - p["T_env"])) / p["C_eff"]
    
    return [dSoC_dt, dT_dt]

# --- 关键：定义 SoC 降为 0 的事件 ---
def soc_depleted_event(t, y, p):
    return y[0]  # 当返回值为 0 时触发

soc_depleted_event.terminal = True  # 触发后停止集成
soc_depleted_event.direction = -1   # 只捕捉从正到负的穿零点

# =========================
# 3. 参数拟合与优化 (反馈模型)
# =========================


# =========================
# 4. 运行仿真并输出结果
# =========================
params = {
    "P1_dl_dbm": 43.0, "P0_dbm": -85.0, "Pmax_dbm": 23.0, "A": 0.45, "alpha_pa": 30.0, "beta_eta": 0.04,
    "B_hz": 10e6, "rho": 0.7, "gamma": 2, "I_down": 0.5, "alpha_cpu": 0.01,
    "P0_cpu": 0.05, "k0": 0.51, "f0": 2.0e9, "u_min": 0.6, "T0_cpu": 45.0, "tau": 5.0,
    "L": 0.3, "S": 0.006, "p": 400, "k1": 0.1025, "k2": 0.0106, "R0": 60.0,
    "eta_0": 0.35, "eta_net": 0.4, "C_eff": 120.0, "T_env": 25.0, "lambda": 0.3,
    "E_base_J": 87240,"eta_dis":0.8 
}

# =========================
# 4. 敏感性分析 (Sensitivity Analysis)
# =========================

def run_simulation(p):
    """
    运行一次仿真，返回电池耗尽时间（小时）。
    如果未耗尽，返回仿真结束时间。
    """
    # 初始温度 25°C, SoC 100%
    sol = solve_ivp(
        battery_ode, 
        [0, 200*3600], # 设置足够长的上限，确保能捕捉到耗尽点
        [1.0, 25.0], 
        args=(p,), 
        events=soc_depleted_event, 
        max_step=60
    )
    
    if sol.t_events[0].size > 0:
        return sol.t_events[0][0] / 3600.0 # 秒转换为小时
    else:
        return sol.t[-1] / 3600.0

target_params = ["gamma", "P0_cpu", "alpha_cpu", "k1", "k2"]
variations_pct = np.linspace(-90, 90, 19) # -90% 到 +90%，共19个点

# 针对每个参数生成一个独立窗口
for param_name in target_params:
    original_val = params[param_name]
    time_results = []
    
    print(f"正在分析参数: {param_name} (原值: {original_val})")
    
    for pct in variations_pct:
        # 计算新参数值：原值 * (1 + 变化率)
        change_rate = pct / 100.0
        new_val = original_val * (1.0 + change_rate)
        
        # 复制参数字典并更新
        current_params = params.copy()
        current_params[param_name] = new_val
        
        # 运行仿真
        t_empty = run_simulation(current_params)
        time_results.append(t_empty)

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(variations_pct, time_results, '*-', linewidth=2, markersize=8)
    
    # 图表装饰
    plt.title(f"Sensitivity Analysis: {param_name}\n(Base Value: {original_val})", fontsize=14)
    plt.xlabel(f"Change in {param_name} (%)", fontsize=12)
    plt.ylabel("Time to Empty (Hour)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置X轴刻度
    plt.xticks(np.arange(-90, 100, 10))
    
    # 添加数值标签 (可选，防止过于密集)
    # for x, y in zip(variations_pct, time_results):
    #     plt.text(x, y, f"{y:.1f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

# 显示所有窗口
plt.show()