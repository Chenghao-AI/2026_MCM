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
    "B_hz": 10e6, "rho": 0.7, "gamma": 2, "I_down": 10, "alpha_cpu": 0.01,
    "P0_cpu": 0.05, "k0": 0.51, "f0": 2.0e9, "u_min": 0.6, "T0_cpu": 45.0, "tau": 5.0,
    "L": 0.3, "S": 0.006, "p": 400, "k1": 0.1025, "k2": 0.0106, "R0": 60.0,
    "eta_0": 0.35, "eta_net": 0.4, "C_eff": 120.0, "T_env": 25.0, "lambda": 0.3,
    "E_base_J": 87240,"eta_dis":0.8 
}

# 执行优化
#tuner = NeuralFeedbackController()
#params = tuner.optimize(base_params)

# 输出优化后的参数
print("\n" + "="*30)
print("优化后的拟合参数:")
print(f"gamma: {params['gamma']:.4f}")
print(f"P0_cpu: {params['P0_cpu']:.4f}")
print(f"alpha_cpu: {params['alpha_cpu']:.4f}")
print(f"k1: {params['k1']:.4f}")
print(f"k2: {params['k2']:.4f}")
print("="*30 + "\n")

try:
    with open("result_params.txt", "w", encoding="utf-8") as f:
        f.write("优化后的拟合参数:\n")
        f.write(f"gamma: {params['gamma']:.4f}\n")
        f.write(f"P0_cpu: {params['P0_cpu']:.4f}\n")
        f.write(f"alpha_cpu: {params['alpha_cpu']:.4f}\n")
        f.write(f"k1: {params['k1']:.4f}\n")
        f.write(f"k2: {params['k2']:.4f}\n")
        f.write(f"Final Ratios: Net={tuner.target_ratios[0]:.2f}, CPU={tuner.target_ratios[1]:.2f}, Dis={tuner.target_ratios[2]:.2f} (Target)\n")
except Exception as e:
    print(f"Write failed: {e}")

# 初始温度 25°C, SoC 100%
sol = solve_ivp(
    battery_ode, 
    [0, 100*3600], # 设置足够长的上限，如 100 小时
    [1.0, 25.0], 
    args=(params,), 
    events=soc_depleted_event, 
    max_step=60
)

# 输出结果
if sol.t_events[0].size > 0:
    end_time_seconds = sol.t_events[0][0]
    print("-" * 30)
    print(f"仿真结果：电池已耗尽")
    print(f"耗时（秒）: {end_time_seconds:.2f} s")
    print(f"耗时（小时）: {end_time_seconds / 3600:.2f} h")
    print("-" * 30)
else:
    print("在预定时间内电池未耗尽。")

# =========================
# 4. 可视化
# =========================

t_h = sol.t / 3600
p_net_v, p_cpu_v, p_dis_v = [], [], []
rsrp_v, snr_v = [], []
for i in range(len(sol.t)):
    p_RF, p_down, p_cpu, p_dis = calculate_all_powers(sol.t[i], sol.y[1,i], params)
    r, s = get_signals_noisy(sol.t[i])
    p_net_v.append(p_RF + p_down); p_cpu_v.append(p_cpu); p_dis_v.append(p_dis)
    rsrp_v.append(r); snr_v.append(s)

fig = plt.figure(figsize=(14, 16))
gs = fig.add_gridspec(5, 4) # 5行4列


# 图 1: 噪声信号
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t_h, rsrp_v, 'g-', alpha=0.6, label='RSRP (dBm)')
ax1_tw = ax1.twinx()
ax1_tw.plot(t_h, snr_v, 'b-', alpha=0.4, label='SNR (dB)')
ax1.set_title(f"1. Real-time Noisy Signals (Total Life: {t_h[-1]:.2f}h)")
ax1.legend(loc='upper left'); ax1_tw.legend(loc='upper right')

# 图 2: 功率演化 (堆叠图)
ax2 = fig.add_subplot(gs[1, :])
# 使用 Matplotlib 默认颜色循环 (Blue, Orange, Green) 以确保统一
colors = ['tab:blue', 'tab:orange', 'tab:green']
ax2.stackplot(t_h, p_net_v, p_cpu_v, p_dis_v, labels=['Network', 'CPU', 'Display'], colors=colors, alpha=0.8)
ax2.set_ylabel("Power (W)")
ax2.set_title("2. Power Component Evolution")
ax2.legend(loc='upper right')

# 图 3: SoC 与 温度
ax3 = fig.add_subplot(gs[2, :])
ax3.plot(t_h, sol.y[0]*100, 'k-', linewidth=2, label='SoC')
ax3_tw = ax3.twinx()
ax3_tw.plot(t_h, sol.y[1], 'r-', alpha=0.8, label='Temp')
ax3.set_ylabel("SoC (%)"); ax3_tw.set_ylabel("Temp (°C)")
ax3.set_title("3. Battery Drain & Thermal Dynamics")

# 图 4: 饼图 (4个时刻)
check_points = [0.1, 0.4, 0.7, 0.95]
for i, q in enumerate(check_points):
    idx = int(len(sol.t) * q)
    ax_pie = fig.add_subplot(gs[3:, i]) # 占用最后两行的每一列
    ax_pie.pie([p_net_v[idx], p_cpu_v[idx], p_dis_v[idx]], 
               labels=['Net', 'CPU', 'Dis'], autopct='%1.1f%%', colors=colors)
    ax_pie.set_title(f"SoC: {sol.y[0,idx]*100:.0f}%\nTime: {t_h[idx]:.1f}h")

plt.tight_layout()
plt.show()