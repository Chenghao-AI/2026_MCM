import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =========================
# 1. 核心数学模型与噪声信号
# =========================
# 修改：去除噪声项，使用固定值，由 params 传入
def get_signals_noisy(t, p=None):
    # 默认值，防止未传入 p 时报错
    if p is None:
        return -95.0, 10.0
    
    # 优先从 p 中获取固定的实验变量
    rsrp = p.get("fixed_rsrp", -95.0)
    snr = p.get("fixed_snr", 10.0)
    
    return float(rsrp), float(snr)

def calculate_all_powers(t, T, p):
    # 修改：传递 p 给 get_signals_noisy
    rsrp, snr = get_signals_noisy(t, p)
    # T0_cpu 为温控保护阈值 (45°C), T 为当前温度
    u = 1.0 if T <= p["T0_cpu"] else (1 - p["u_min"]) * np.exp(-(T - p["T0_cpu"]) / p["tau"]) + p["u_min"]
    
    p_b = 0.0

    if p.get("mode") == "standby":
        p_RF = 0.0
        p_down = 0.0
        p_dis = 0.0
        # CPU & Display
        p_cpu = (p["P0_cpu"] + (p["k0"] + p["alpha_cpu"] * p["I_down"]) * (p["f0"] * u / 1e9))/p["eta_0"]
        # Network power for standby
        p_b = (p["gamma"] * p["I_down"] + p["f_b"] * p["PE_0"]) / p["eta_0"]
    else:
        # Network
        PL = p["P1_dl_dbm"] - rsrp
        Pup_W = 10**((np.minimum(p["P0_dbm"] + PL, p["Pmax_dbm"]) - 30)/10)
        eta = 0.4
        v_down = p["rho"] * p["B_hz"] * np.log2(1 + 10**(snr/10))
        p_down = ((p["gamma"] * p["I_down"]) )/p["eta_net"]
        p_RF = ((Pup_W / eta) / (v_down / 1e6) * p["I_down"])

        # CPU & Display
        p_cpu = (p["P0_cpu"] + (p["k0"] + p["alpha_cpu"] * p["I_down"]) * (p["f0"] * u / 1e9))/p["eta_0"]
        p_dis = ((p["k1"] * p["L"] + p["k2"] * p["R0"] * u) * p["S"] * p["p"])/p["eta_dis"]
    
    return p_RF, p_down, p_cpu, p_dis, p_b

# =========================
# 2. ODE 定义与事件检测
# =========================
def battery_ode(t, y, p):
    SoC, T = y
    p_RF, p_down, p_cpu, p_dis, p_b = calculate_all_powers(t, T, p)
    P_tot = p_RF + p_down + p_cpu + p_dis + p_b
    
    # 电池容量修正
    # 限制温度以防止在高温下容量计算发散
    T_cal = np.minimum(T, 60.0)
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
# 3. 运行控制变量实验
# =========================

# 基础参数配置（包含用户指定的固定参数）
base_params = {
    "P1_dl_dbm": 43.0, "P0_dbm": -85.0, "Pmax_dbm": 23.0, "A": 0.45, "alpha_pa": 30.0, "beta_eta": 0.04,
    "B_hz": 10e7, "rho": 0.7, 
    # 固定参数
    "gamma": 2.0000, 
    "P0_cpu": 0.0500,
    "alpha_cpu": 0.0100,
    "k1": 0.1025,
    "k2": 0.0106,
    
    "I_down": 0.5, "k0": 0.51, "f0": 2.0e9, "u_min": 0.6, "T0_cpu": 45.0, "tau": 5.0,
    "L": 0.3, "S": 0.006, "p": 400, "R0": 60.0,
    "eta_0": 0.35, "eta_net": 0.4, "C_eff": 120.0, "T_env": 25.0, "lambda": 0.3,
    "E_base_J": 87240, "eta_dis": 0.8,
    "f_b": 0.02948, "PE_0": 150/1000
}

def run_simulation(p):
    sol = solve_ivp(
        battery_ode, 
        [0, 200*3600], # 上限足够大
        [1.0, 25.0], 
        args=(p,), 
        events=soc_depleted_event, 
        max_step=60
    )
    if sol.t_events[0].size > 0:
        return sol.t_events[0][0] / 3600.0 # 返回小时
    else:
        return sol.t[-1] / 3600.0 # 未耗尽

# =========================
# 3. 运行控制变量实验 - 手机使用模式对电池寿命的影响
# =========================
print("开始实验: 手机使用模式对电池 Time to Empty 的影响...")

modes = ["game", "video", "word", "standby"]
time_to_empty_results = []

for mode in modes:
    p = base_params.copy()
    p["mode"] = mode
    
    if mode == "game":
        p["k0"] = 0.51
        p["I_down"] =0.5
    elif mode == "video":
        p["k0"] = 0.32
        p["I_down"] = 1
    elif mode == "word":
        p["k0"] = 0.21
        p["I_down"] = 0.2
    elif mode == "standby":
        p["k0"] = 0.16
        p["I_down"] = 0.01
        
    time_val = run_simulation(p)
    time_to_empty_results.append(time_val)
    print(f"Mode: {mode}, Time to Empty: {time_val:.4f} hours")

# =========================
# 4. 可视化结果
# =========================
plt.figure(figsize=(10, 6))
bars = plt.bar(modes, time_to_empty_results, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'])

plt.title("Impact of Usage Mode on Battery Life")
plt.xlabel("Usage Mode")
plt.ylabel("Time to Empty (hours)")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f} h',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()