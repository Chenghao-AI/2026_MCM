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
    p_b = (p["gamma"] * p["I_down"] + p["f_b"] * p["PE_0"]) / p["eta_0"]
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
# 3. 比例拟合控制器
# =========================
class RatioFittingController:
    def __init__(self, target_ratio=12.0):
        self.target_ratio = target_ratio
        self.lr = 0.001 
        
    def optimize(self, params):
        print(f"  -> 开始拟合 (PE_0={params['PE_0']})...")
        current_p = params.copy()
        current_p["f_b"] = 0.01 
        
        for i in range(200):
            p_RF, _, p_cpu, _ = calculate_all_powers(0, 25.0, current_p)
            if p_cpu == 0: break
            
            current_ratio = p_RF / p_cpu
            error = current_ratio - self.target_ratio
            
            # 更新 f_b
            current_p["f_b"] -= error * 0.001 * self.lr * 100
            if current_p["f_b"] < 1e-6: current_p["f_b"] = 1e-6
        
        p_RF, _, p_cpu, _ = calculate_all_powers(0, 25.0, current_p)
        print(f"  -> 拟合结束: f_b={current_p['f_b']:.6f}, Net/CPU Ratio={p_RF/p_cpu:.2f}")
        return current_p

# =========================
# 4. 封装绘图函数
# =========================
def run_and_plot(params, scenario_name):
    print("\n" + "="*50)
    print(f"Running Scenario: {scenario_name}")
    print("="*50)
    
    # 1. 拟合
    tuner = RatioFittingController(target_ratio=12.0)
    optimized_params = tuner.optimize(params)
    
    # 2. 仿真
    sol = solve_ivp(
        battery_ode, [0, 100*3600], [1.0, 25.0], 
        args=(optimized_params,), events=soc_depleted_event, max_step=60
    )
    
    # 3. 数据处理
    t_h = sol.t / 3600
    p_net_v, p_cpu_v = [], []
    rsrp_v, snr_v = [], []

    for i in range(len(sol.t)):
        p_RF, p_down, p_cpu, p_dis = calculate_all_powers(sol.t[i], sol.y[1,i], optimized_params)
        r, s = get_signals_noisy(sol.t[i])
        p_net_v.append(p_RF)
        p_cpu_v.append(p_cpu)
        rsrp_v.append(r); snr_v.append(s)
        
    # 4. 绘图
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(5, 4)
    
    # Title
    plt.suptitle(f"Scenario: {scenario_name} (PE_0={optimized_params['PE_0']}, f_b={optimized_params['f_b']:.5f})", fontsize=16, weight='bold')

    # Plot 1: Signals
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_h, rsrp_v, 'g-', alpha=0.6, label='RSRP')
    ax1_tw = ax1.twinx()
    ax1_tw.plot(t_h, snr_v, 'b-', alpha=0.4, label='SNR')
    ax1.set_title("1. Signal Environment")
    ax1.legend(loc='upper left'); ax1_tw.legend(loc='upper right')

    # Plot 2: Stackplot (Net vs CPU)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.stackplot(t_h, p_net_v, p_cpu_v, labels=['Network', 'CPU'], colors=['tab:blue', 'tab:orange'], alpha=0.8)
    ax2.set_ylabel("Power (W)")
    ax2.set_title("2. Power Consumption (Display=0)")
    ax2.legend(loc='upper right')

    # Plot 3: SoC
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(t_h, sol.y[0]*100, 'k-', linewidth=2, label='SoC')
    ax3_tw = ax3.twinx()
    ax3_tw.plot(t_h, sol.y[1], 'r-', alpha=0.8, label='Temp')
    ax3.set_ylabel("SoC (%)"); ax3_tw.set_ylabel("Temp (°C)")
    ax3.set_title(f"3. Battery & Thermal (Life: {t_h[-1]:.2f}h)")

    # Plot 4: Pie Charts
    check_points = [0.1, 0.4, 0.7, 0.95]
    for i, q in enumerate(check_points):
        idx = int(len(sol.t) * q)
        if idx >= len(sol.t): idx = len(sol.t) - 1
        
        ax_pie = fig.add_subplot(gs[3:, i]) 
        sizes = [p_net_v[idx], p_cpu_v[idx]]
        ratio = sizes[0]/sizes[1] if sizes[1] > 0 else 0
        
        ax_pie.pie(sizes, labels=['Net', 'CPU'], autopct='%1.1f%%', colors=['tab:blue', 'tab:orange'], startangle=90)
        ax_pie.set_title(f"SoC: {sol.y[0,idx]*100:.0f}%\nRatio: {ratio:.1f}:1")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =========================
# 5. 主程序执行
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
    "f_b": 0.0 # 初始
}

# --- 运行场景 : Cellular ---
p1 = base_params.copy()
p1["PE_0"] = 150
run_and_plot(p1, "Cellular Mode (PE_0 = 150)")