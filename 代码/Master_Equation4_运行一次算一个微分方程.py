import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 设置全局绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False 
plt.style.use('seaborn-v0_8-muted') # 使用更现代的配色风格

# =========================
# 1. 动态输入生成 (信号与业务量)
# =========================
def get_user_behavior(t):
    behavior_sequence = [0.5, 5.0, 20.0, 2.0, 15.0, 0.2, 8.0, 1.0]
    interval_idx = int((t // 1800) % len(behavior_sequence))
    expected_i = behavior_sequence[interval_idx]
    seed_val = int(abs(t * 100)) % 10**8
    rng = np.random.default_rng(seed_val)
    noise = rng.normal(0, expected_i * 0.1)
    return float(np.maximum(0.1, expected_i + noise))

def get_signals_noisy(t):
    rsrp_base = -95 + 15 * np.sin(2 * np.pi * t / 3600)
    snr_base = 10 + 8 * np.cos(2 * np.pi * t / 1800)
    seed_val = int(abs(t * 100)) % 10**8
    rng = np.random.default_rng(seed_val)
    shadowing = 3.0 * np.sin(2 * np.pi * t / 600) 
    rsrp = np.clip(rsrp_base + shadowing + rng.normal(0, 1.5), -120, -60)
    snr = np.clip(snr_base + shadowing*0.5 + rng.normal(0, 2.0), -2, 28)
    return float(rsrp), float(snr)

# =========================
# 2. 核心数学模型函数
# =========================
def calculate_all_powers(t, T, p):
    rsrp, snr = get_signals_noisy(t)
    i_down = get_user_behavior(t)
    u = 1.0 if T <= p["T0_cpu"] else (1 - p["u_min"]) * np.exp(-(T - p["T0_cpu"]) / p["tau"]) + p["u_min"]
    
    PL = p["P1_dl_dbm"] -rsrp
    P_PA=np.minimum(p["P0_dbm"] + PL, p["Pmax_dbm"])
    Pup_W = 10**((np.minimum(p["P0_dbm"] - PL, p["Pmax_dbm"]) - 30)/10)
    eta = 0.4
    v_down = p["rho"] * p["B_hz"] * np.log2(1 + 10**(snr/10))
    p_down = (p["gamma"] * i_down) 
    p_RF = ((Pup_W / eta) / (v_down / 1e6) * i_down)
    
    p_cpu = p["P0_cpu"] + (p["k0"] + p["alpha_cpu"] * i_down) * (p["f0"] * u / 1e9)
    p_dis = (p["k1"] * p["L"] + p["k2"] * p["R0"] * u) * p["S"] * p["p"]
    
    return p_RF, p_down, p_cpu, p_dis, i_down, eta, rsrp, snr

def battery_ode(t, y, p):
    SoC, T = y
    if SoC <= 0: return [0, 0]
    p_RF, p_down, p_cpu, p_dis, _, eta, _, _ = calculate_all_powers(t, T, p)
    P_tot = p_RF + p_down + p_cpu + p_dis
    
    E_joules = p["E_base_J"] * (13.696*T**3 - 424.29*T**2 + 3588.6*T + 1862.4) / 106000
    dSoC_dt = -P_tot / E_joules
    dT_dt = ((1-p["eta_0"])*(p_cpu+p_dis+p_down) + (1-eta)*p_RF - p["lambda"]*(T - p["T_env"])) / p["C_eff"]
    return [dSoC_dt, dT_dt]

def simulate_battery(params, SoC0=1.0, T0=25.0, max_time_h=24):
    def soc_event(t, y, p): return y[0]
    soc_event.terminal = True
    soc_event.direction = -1
    return solve_ivp(fun=battery_ode, t_span=[0, max_time_h * 3600], y0=[SoC0, T0], args=(params,), events=soc_event, max_step=30)

# =========================
# 3. 执行与可视化
# =========================
if __name__ == "__main__":
    my_params = {
        "P1_dl_dbm": 43.0, "P0_dbm": -85.0, "Pmax_dbm": 23.0, 
        "B_hz": 10e6, "rho": 0.7, "gamma": 0.15, "alpha_cpu": 0.03,
        "P0_cpu": 0.5, "k0": 1.2, "f0": 2.0e9, "u_min": 0.6, "T0_cpu": 45.0, "tau": 5.0,
        "L": 0.6, "S": 0.006, "p": 400, "k1": 0.8, "k2": 0.02, "R0": 60.0,
        "eta_0": 0.7, "C_eff": 150.0, "T_env": 25.0, "lambda": 0.35,
        "E_base_J": 4500 * 3.8 * 3.6
    }

    res = simulate_battery(my_params)
    t_h = res.t / 3600
    SoC, T = res.y[0], res.y[1]
    
    p_net_v, p_cpu_v, p_dis_v, i_down_v, rsrp_v, snr_v = [], [], [], [], [], []
    for i in range(len(res.t)):
        pRF, pdown, pc, pd, idwn, eta, rsrp, snr = calculate_all_powers(res.t[i], T[i], my_params)
        p_net_v.append(pRF + pdown); p_cpu_v.append(pc); p_dis_v.append(pd)
        i_down_v.append(idwn); rsrp_v.append(rsrp); snr_v.append(snr)

    print(f"电池耗尽总时长: {t_h[-1]:.2f} 小时")
    print(f"RF效率:{eta}")
    print(f"{pRF/(pRF+pdown)}")
    # --- 第一张图：信号环境与功率演化 ---
    fig1, (ax0, ax_net, ax1) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # 1. 业务量
    ax0.step(t_h, i_down_v, where='post', color='#34495e', alpha=0.8, label='Data Rate (Mbps)')
    ax0.fill_between(t_h, i_down_v, step="post", alpha=0.1, color='#34495e')
    ax0.set_ylabel("Rate (Mbps)")
    ax0.set_title("I. User Behavior Data Rate", loc='left', fontweight='bold')
    ax0.grid(True, alpha=0.2)

    # 2. 信号图 (代替原来 SoC 的位置)
    ax_net.plot(t_h, rsrp_v, color='#27ae60', linewidth=1.2, label='RSRP (dBm)')
    ax_net.set_ylabel("RSRP (dBm)", color='#27ae60')
    ax_net.tick_params(axis='y', labelcolor='#27ae60')
    ax_net_snr = ax_net.twinx()
    ax_net_snr.plot(t_h, snr_v, color='#8e44ad', linewidth=1.0, alpha=0.7, label='SNR (dB)')
    ax_net_snr.set_ylabel("SNR (dB)", color='#8e44ad')
    ax_net_snr.tick_params(axis='y', labelcolor='#8e44ad')
    ax_net.set_title("II. Network Environment (RSRP & SNR)", loc='left', fontweight='bold')
    ax_net.grid(True, alpha=0.2, linestyle='--')

    # 3. 功率演化
    ax1.stackplot(t_h, p_net_v, p_cpu_v, p_dis_v, labels=['Network', 'CPU', 'Display'], 
                  colors=['#3498db', '#e67e22', '#2ecc71'], alpha=0.8)
    ax1.set_ylabel("Power (W)")
    ax1.set_xlabel("Time (hours)")
    ax1.set_title("III. Power Component Evolution", loc='left', fontweight='bold')
    ax1.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()

    # --- 第二张图：SoC 斜率分析与饼图 ---
    fig2 = plt.figure(figsize=(14, 11))
    gs = fig2.add_gridspec(3, 4)

    # 1. SoC 曲线 (调整 y 轴使斜率明显)
    ax_soc = fig2.add_subplot(gs[0, :])
    ax_soc.plot(t_h, SoC * 100, color='#2980b9', linewidth=2.5, label='SoC')
    # 动态调整 y 轴刻度，聚焦在 0-100 范围，并增加次级网格
    ax_soc.set_ylim(-2, 102)
    ax_soc.set_ylabel("State of Charge (%)", fontweight='bold')
    ax_soc.set_title("IV. Battery SoC Drainage (Slope Analysis)", loc='left', fontweight='bold')
    ax_soc.grid(True, which='major', alpha=0.4)
    ax_soc.grid(True, which='minor', alpha=0.1, linestyle=':')
    ax_soc.minorticks_on()

    # 2. 温度曲线
    ax_temp = fig2.add_subplot(gs[1, :], sharex=ax_soc)
    ax_temp.plot(t_h, T, color='#e74c3c', linewidth=2, label='Temperature')
    ax_temp.axhline(my_params["T0_cpu"], color='#7f8c8d', linestyle='--', label='Throttling')
    ax_temp.set_ylabel("Temp (°C)", fontweight='bold')
    ax_temp.set_xlabel("Time (hours)")
    ax_temp.legend(loc='lower right')
    ax_temp.grid(True, alpha=0.3)

    # 3. 饼图
    check_points = [0.1, 0.4, 0.7, 0.95]
    pie_colors = ['#3498db', '#e67e22', '#2ecc71']
    for i, q in enumerate(check_points):
        ax_pie = fig2.add_subplot(gs[2, i])
        idx = int(len(t_h) * q)
        if idx < len(t_h):
            powers = [p_net_v[idx], p_cpu_v[idx], p_dis_v[idx]]
            ax_pie.pie(powers, labels=['Net', 'CPU', 'Dis'], autopct='%1.1f%%', 
                       colors=pie_colors, startangle=140, pctdistance=0.75,
                       explode=[0.05, 0, 0], textprops={'fontsize': 9})
            ax_pie.set_title(f'T={t_h[idx]:.1f}h | SoC={SoC[idx]*100:.0f}%', fontsize=10, pad=10)

    fig2.suptitle(f"System Efficiency & Drainage Analysis (Total Life: {t_h[-1]:.2f}h)", 
                  fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()