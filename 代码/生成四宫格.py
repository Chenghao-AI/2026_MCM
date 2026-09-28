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
    # T0_cpu 为温控保护阈值 (45°C), T 为当前温度
    u = 1.0 if T <= p["T0_cpu"] else (1 - p["u_min"]) * np.exp(-(T - p["T0_cpu"]) / p["tau"]) + p["u_min"]
    
    # 特殊处理 Standby 模式
    if p.get("mode") == "Standby":
        # --- CPU Power ---
        p_cpu = (p["P0_cpu"] + (p["k0"] + p["alpha_cpu"] * p["I_down"]) * (p["f0"] * u / 1e9))/p["eta_0"]
        
        # --- Network Power (p_b) ---
        # 核心公式: p_b = (gamma * I_down + f_b * PE_0) / eta_0
        # f_b = 0.02948, PE_0 = 0.15
        p_b = (p["gamma"] * p["I_down"] + 0.02948 * 0.15) / p["eta_0"]
        p_RF = p_b
        p_down = 0.0
        
        # --- Display Power ---
        p_dis = 0.0
        
        return p_RF, p_down, p_cpu, p_dis

    # 其他模式 (Game, Video, Word)
    rsrp, snr = get_signals_noisy(t)
    
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
# 4. 运行仿真并输出结果 (四模式对比)
# =========================

# 基础参数配置
base_params = {
    "P1_dl_dbm": 43.0, "P0_dbm": -85.0, "Pmax_dbm": 23.0, "A": 0.45, "alpha_pa": 30.0, "beta_eta": 0.04,
    "B_hz": 10e6, "rho": 0.7, "gamma": 2, "I_down": 1, "alpha_cpu": 0.01,
    "P0_cpu": 0.05, "k0": 0.32, "f0": 2.0e9, "u_min": 0.6, "T0_cpu": 45.0, "tau": 5.0,
    "L": 0.3, "S": 0.006, "p": 400, "k1": 0.1025, "k2": 0.0106, "R0": 60.0,
    "eta_0": 0.35, "eta_net": 0.4, "C_eff": 120.0, "T_env": 25.0, "lambda": 0.3,
    "E_base_J": 87240, "eta_dis": 0.8 
}

# 四种模式配置
modes = {
    "Game": {"k0": 0.51, "I_down": 0.5},
    "Video": {"k0": 0.32, "I_down": 1.0},
    "Word": {"k0": 0.21, "I_down": 0.2},
    "Standby": {"k0": 0.16, "I_down": 0.01}
}

fig, axes = plt.subplots(2, 2, figsize=(10, 10)) # 2x2 grid, roughly square window
axes_flat = axes.flatten()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Network, CPU, Display

print("开始仿真四种模式...")

for i, (mode_name, mode_params) in enumerate(modes.items()):
    # 复制参数并更新当前模式参数
    p = base_params.copy()
    p.update(mode_params)
    p["mode"] = mode_name # 重要：设置模式名称以便 calculate_all_powers 识别
    
    # 运行仿真
    sol = solve_ivp(
        battery_ode, 
        [0, 100*3600], 
        [1.0, 25.0], 
        args=(p,), 
        events=soc_depleted_event, 
        max_step=60
    )
    
    ax = axes_flat[i]
    
    # 获取耗尽前一刻的索引
    if sol.t_events[0].size > 0:
        idx = -1 # 最后一个点
        end_time_h = sol.t[idx] / 3600
        print(f"Mode: {mode_name:<8} | Time to Empty: {end_time_h:.2f} h")
        
        # 计算该时刻功率
        p_RF, p_down, p_cpu, p_dis = calculate_all_powers(sol.t[idx], sol.y[1, idx], p)
        p_net = p_RF + p_down
        
        # 绘制饼图
        if mode_name == "Standby":
            # Standby 模式不显示 Display (因为是 0)
            ax.pie([p_net, p_cpu], 
                   labels=['Net', 'CPU'], 
                   autopct='%1.1f%%', 
                   colors=[colors[0], colors[1]], # 只取前两个颜色
                   startangle=90)
        else:
            # 其他模式正常显示三项
            ax.pie([p_net, p_cpu, p_dis], 
                   labels=['Net', 'CPU', 'Dis'], 
                   autopct='%1.1f%%', 
                   colors=colors,
                   startangle=90)
                   
        ax.set_title(f"{mode_name}\n(End Time: {end_time_h:.1f}h)")
    else:
        print(f"Mode: {mode_name:<8} | Battery not depleted")
        ax.text(0.5, 0.5, "Not Depleted", ha='center')

plt.tight_layout()
plt.show()