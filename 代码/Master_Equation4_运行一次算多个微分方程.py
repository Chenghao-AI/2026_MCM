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
def get_user_behavior(t, seed_offset=0):
    behavior_sequence = [0.5, 5.0, 20.0, 2.0, 15.0, 0.2, 8.0, 1.0]
    interval_idx = int((t // 1800) % len(behavior_sequence))
    expected_i = behavior_sequence[interval_idx]
    # 引入 seed_offset 以支持多次实验的随机性差异
    seed_val = int(abs(t * 100) + seed_offset) % 10**8
    rng = np.random.default_rng(seed_val)
    noise = rng.normal(0, expected_i * 0.1)
    return float(np.maximum(0.1, expected_i + noise))

def get_signals_noisy(t, seed_offset=0):
    rsrp_base = -95 + 15 * np.sin(2 * np.pi * t / 3600)
    snr_base = 10 + 8 * np.cos(2 * np.pi * t / 1800)
    # 引入 seed_offset 以支持多次实验的随机性差异
    seed_val = int(abs(t * 100) + seed_offset) % 10**8
    rng = np.random.default_rng(seed_val)
    shadowing = 3.0 * np.sin(2 * np.pi * t / 600) 
    rsrp = np.clip(rsrp_base + shadowing + rng.normal(0, 1.5), -120, -60)
    snr = np.clip(snr_base + shadowing*0.5 + rng.normal(0, 2.0), -2, 28)
    return float(rsrp), float(snr)

# =========================
# 2. 核心数学模型函数
# =========================
def calculate_all_powers(t, T, p):
    # 从参数中获取 seed_offset，默认为0
    seed_offset = p.get("seed_offset", 0)
    rsrp, snr = get_signals_noisy(t, seed_offset)
    i_down = get_user_behavior(t, seed_offset)
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
# 3. 执行与可视化 (Monte Carlo 体系化实验)
# =========================
def run_monte_carlo_simulation(params, n_simulations=100):
    print(f"开始进行 {n_simulations} 次 Monte Carlo 模拟...")
    
    # 统一的时间轴用于插值统计 (0到24小时, 1000个点)
    common_time_h = np.linspace(0, 24, 1000)
    all_soc_series = []
    battery_lives = []
    
    # 存储最后一次运行的详细数据用于示例展示
    last_run_data = None
    
    for i in range(n_simulations):
        # 为每次模拟设置不同的随机种子偏移量
        current_params = params.copy()
        current_params["seed_offset"] = np.random.randint(0, 100000) + i * 1000
        
        # 运行单次模拟
        res = simulate_battery(current_params)
        
        # 记录电池寿命 (小时)
        life_h = res.t[-1] / 3600
        battery_lives.append(life_h)
        
        # 插值 SoC 到统一时间轴
        # 如果模拟提前结束(SoC=0), interp 会自动用最后一个值(0)填充后续时间
        soc_interp = np.interp(common_time_h, res.t / 3600, res.y[0])
        all_soc_series.append(soc_interp)
        
        # 保存最后一次运行的数据用于详细展示
        if i == n_simulations - 1:
            last_run_data = {
                "res": res,
                "params": current_params
            }
        
        # 简单的进度打印
        if (i+1) % 5 == 0:
            print(f"  已完成 {i+1}/{n_simulations} 次模拟")
            
    # 转换为 numpy 数组方便计算
    all_soc_matrix = np.array(all_soc_series)
    
    # 计算均值和标准差
    mean_soc = np.mean(all_soc_matrix, axis=0)
    std_soc = np.std(all_soc_matrix, axis=0)
    
    results = {
        "time_h": common_time_h,
        "mean_soc": mean_soc,
        "std_soc": std_soc,
        "all_soc": all_soc_matrix,
        "battery_lives": np.array(battery_lives),
        "last_run": last_run_data
    }
    
    return results

def plot_monte_carlo_results(results):
    t_h = results["time_h"]
    mean_soc = results["mean_soc"]
    std_soc = results["std_soc"]
    lives = results["battery_lives"]
    
    # --- 图1: SoC 统计分析 (均值 + 方差) ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # 绘制所有单次轨迹(细线, 透明)
    # 增加透明度并添加图例标签，仅标记第一条以避免图例重复
    for i, soc_trace in enumerate(results["all_soc"]):
        label = 'Individual Runs' if i == 0 else None
        ax1.plot(t_h, soc_trace * 100, color='gray', alpha=0.3, linewidth=0.8, label=label)
        
    # 绘制均值曲线
    ax1.plot(t_h, mean_soc * 100, color='#2980b9', linewidth=2.5, label='Mean SoC')
    
    # 绘制方差范围 (Mean ± Std)
    ax1.fill_between(t_h, (mean_soc - std_soc) * 100, (mean_soc + std_soc) * 100, 
                     color='#2980b9', alpha=0.3, label='Standard Deviation (±1σ)')
    
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("State of Charge (%)")
    ax1.set_title(f"Monte Carlo Analysis: Battery SoC Evolution ({len(lives)} Runs)", fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_ylim(-2, 102)
    
    # --- 图2: 电池寿命分布直方图 ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax2.hist(lives, bins=15, color='#27ae60', alpha=0.7, rwidth=0.85, edgecolor='black')
    
    # 添加平均值线
    mean_life = np.mean(lives)
    life_variance = np.var(lives) # 计算寿命方差
    std_life = np.std(lives)
    
    ax2.axvline(mean_life, color='#c0392b', linestyle='--', linewidth=2, label=f'Mean Life: {mean_life:.2f} h')
    
    ax2.set_xlabel("Battery Life (hours)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Battery Life", fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    print(f"平均电池寿命: {mean_life:.4f} 小时")
    print(f"寿命标准差: {std_life:.6f} 小时")
    print(f"寿命方差: {life_variance:.6e} (小时^2)")
    print(f"最短寿命: {np.min(lives):.4f} 小时")
    print(f"最长寿命: {np.max(lives):.4f} 小时")
    
    # --- 图3: 单次典型运行的详细物理量 (复用之前的可视化逻辑) ---
    # 使用最后一次运行的数据作为示例
    last_data = results["last_run"]
    res = last_data["res"]
    params = last_data["params"]
    
    # 重新计算该次运行的详细物理量
    res_t_h = res.t / 3600
    res_T = res.y[1]
    p_net_v, p_cpu_v, p_dis_v, i_down_v, rsrp_v, snr_v = [], [], [], [], [], []
    
    for i in range(len(res.t)):
        pRF, pdown, pc, pd, idwn, eta, rsrp, snr = calculate_all_powers(res.t[i], res_T[i], params)
        p_net_v.append(pRF + pdown); p_cpu_v.append(pc); p_dis_v.append(pd)
        i_down_v.append(idwn); rsrp_v.append(rsrp); snr_v.append(snr)
        
    fig3, (ax_rate, ax_env, ax_pow) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # 1. 业务量
    ax_rate.step(res_t_h, i_down_v, where='post', color='#34495e', alpha=0.8, label='Data Rate')
    ax_rate.set_ylabel("Rate (Mbps)")
    ax_rate.set_title("Single Run Detail: User Behavior & Environment", loc='left', fontweight='bold')
    ax_rate.grid(True, alpha=0.2)
    
    # 2. 信号环境
    ax_env.plot(res_t_h, rsrp_v, color='#27ae60', label='RSRP')
    ax_env.set_ylabel("RSRP (dBm)", color='#27ae60')
    ax_env_snr = ax_env.twinx()
    ax_env_snr.plot(res_t_h, snr_v, color='#8e44ad', alpha=0.7, label='SNR')
    ax_env_snr.set_ylabel("SNR (dB)", color='#8e44ad')
    ax_env.grid(True, alpha=0.2)
    
    # 3. 功率堆叠
    ax_pow.stackplot(res_t_h, p_net_v, p_cpu_v, p_dis_v, labels=['Network', 'CPU', 'Display'],
                     colors=['#3498db', '#e67e22', '#2ecc71'], alpha=0.8)
    ax_pow.set_ylabel("Power (W)")
    ax_pow.set_xlabel("Time (hours)")
    ax_pow.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    my_params = {
        "P1_dl_dbm": 43.0, "P0_dbm": -85.0, "Pmax_dbm": 23.0, 
        "B_hz": 10e6, "rho": 0.7, "gamma": 0.15, "alpha_cpu": 0.03,
        "P0_cpu": 0.5, "k0": 1.2, "f0": 2.0e9, "u_min": 0.6, "T0_cpu": 45.0, "tau": 5.0,
        "L": 0.6, "S": 0.006, "p": 400, "k1": 0.8, "k2": 0.02, "R0": 60.0,
        "eta_0": 0.7, "C_eff": 150.0, "T_env": 25.0, "lambda": 0.35,
        "E_base_J": 4500 * 3.8 * 3.6
    }

    # 执行 Monte Carlo 模拟
    mc_results = run_monte_carlo_simulation(my_params, n_simulations=100)
    
    # 可视化结果
    plot_monte_carlo_results(mc_results)