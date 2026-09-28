import subprocess
import time
import csv
import re
import sys
import os
from datetime import datetime

# ===================== 配置区（可修改） =====================
SAMPLING_INTERVAL = 2  # 采样间隔（秒），建议2~5秒
DEFAULT_AMBIENT_TEMP = 25.0    # 默认环境温度（手动修正基准）
# 电池额定容量 (mAh)，用于库仑计估算。红米K80系列通常较大，假设 6000mAh
BATTERY_CAPACITY_MAH = 6000 
# ============================================================

def get_output_filename():
    """生成带时间戳的文件名，避免权限冲突"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"MCM_Battery_Data_{timestamp}.csv"

OUTPUT_FILE = get_output_filename()

def run_adb_cmd(cmd, timeout=5):
    """执行ADB命令，返回输出结果（处理超时/异常）"""
    full_cmd = f"adb shell {cmd}"
    try:
        # 使用utf-8解码，忽略错误防止特殊字符导致崩溃
        result = subprocess.run(
            full_cmd, shell=True, capture_output=True, text=True, 
            timeout=timeout, encoding='utf-8', errors='ignore'
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Timeout"
    except Exception as e:
        return f"Error: {str(e)}"

def get_ambient_temp():
    """
    全网扫描模式：尝试获取手机内部温度传感器（如 skin/case）作为环境温度参考。
    """
    try:
        # 1. 优先尝试直接扫描 sysfs 节点
        sys_paths = run_adb_cmd("ls /sys/class/thermal/thermal_zone*/type").split()
        
        target_zones = []
        for path in sys_paths:
            if not path.strip(): continue
            type_name = run_adb_cmd(f"cat {path}")
            type_name = type_name.lower()
            
            # 关键词匹配优先级
            if "skin" in type_name or "case" in type_name:
                target_zones.append((path.replace("type", "temp"), 1))
            elif "therm" in type_name or "chg" in type_name or "qi" in type_name:
                target_zones.append((path.replace("type", "temp"), 2))
        
        target_zones.sort(key=lambda x: x[1])
        
        for temp_path, _ in target_zones:
            temp_str = run_adb_cmd(f"cat {temp_path}")
            if temp_str.isdigit():
                temp_val = int(temp_str)
                if temp_val > 1000:
                    return temp_val / 1000.0
                elif 0 < temp_val < 100:
                    return float(temp_val)

        # 2. 回退到 dumpsys
        dump = run_adb_cmd("dumpsys thermalservice")
        skin_match = re.search(r"Temperature\{mValue=(\d+\.?\d*),\s*mType=(?:SKIN|11|3)", dump, re.IGNORECASE)
        if not skin_match:
             skin_match = re.search(r"Temperature\{mValue=(\d+\.?\d*),\s*[^}]*skin", dump, re.IGNORECASE)
             
        if skin_match:
            temp = float(skin_match.group(1))
            if 0 < temp < 80:
                return temp
    except Exception:
        pass
    return DEFAULT_AMBIENT_TEMP

# 全局变量，用于库仑计估算 SOC
last_charge_counter = None
last_check_time = None
calibrated_capacity_mah = BATTERY_CAPACITY_MAH
is_calibrated = False

# 脉冲更新相关状态
REFRESH_INTERVAL = 45 # 多少秒刷新一次硬件 (Reset)
last_refresh_time = 0
estimated_current_ma = 200.0 # 初始假设电流，用于 Dead Reckoning
last_real_cc = None # 上次从硬件读到的真实 CC

def get_battery_core_data():
    """获取电池SOC、温度、电压、电流（核心因变量）"""
    global last_charge_counter, last_check_time, calibrated_capacity_mah, is_calibrated
    global last_refresh_time, estimated_current_ma, last_real_cc
    
    data = {"SOC": None, "Battery_Temp": None, "Voltage": None, "Current": 0}
    now = time.time()
    
    # ---------------------------------------------------------
    # 策略选择：Pulse Update (脉冲更新)
    # ---------------------------------------------------------
    should_refresh = (now - last_refresh_time) > REFRESH_INTERVAL
    
    if should_refresh:
        # 1. 强制刷新：Reset 电池服务，让数据动起来
        run_adb_cmd("dumpsys battery reset")
        time.sleep(1.5) # 给系统 1.5s 反应时间 (越短越好，减少充电影响)
        
        # 2. 读取真实数据
        battery_dump = run_adb_cmd("dumpsys battery")
        
        # 3. 立即重新冻结 (set usb 0)
        run_adb_cmd("dumpsys battery set usb 0")
        
        last_refresh_time = now
    else:
        # 盲区模式：不读取 dumpsys，因为读了也是死值
        # 仅读取温度 (因为走的是 sysfs，不受影响)
        battery_dump = "" 
    
    # === 1. 温度 (始终尝试读取硬件) ===
    raw_temp = None
    try:
        sys_paths = run_adb_cmd("ls /sys/class/thermal/thermal_zone*/type").split()
        for path in sys_paths:
            if not path.strip(): continue
            type_name = run_adb_cmd(f"cat {path}").lower()
            if "battery" in type_name or "batt" in type_name:
                temp_path = path.replace("type", "temp")
                val = run_adb_cmd(f"cat {temp_path}")
                if val.isdigit():
                    raw_temp = int(val)
                    break
    except: pass
    
    if raw_temp is not None:
        data["Battery_Temp"] = raw_temp / 1000.0 if raw_temp > 1000 else raw_temp / 10.0
    else:
        # 如果是刷新时刻，可以用 dumpsys 的值兜底
        if should_refresh:
            temp_match = re.search(r"(?:temperature|batt_temp):\s*(\d+)", battery_dump, re.IGNORECASE)
            data["Battery_Temp"] = int(temp_match.group(1))/10.0 if temp_match else 0
        else:
            data["Battery_Temp"] = 0 # 盲区如果不读硬件，只能给0，或者保持上一次的值？
            # 改进：保持上一次的值太麻烦，暂时给0，或者希望 sysfs 能读到
    
    # === 2. SOC & Current (核心算法) ===
    if should_refresh:
        # 解析真实 CC
        cc_match = re.search(r"Charge counter:\s*(\d+)", battery_dump, re.IGNORECASE)
        current_cc = int(cc_match.group(1)) if cc_match else None
        
        # 解析 System SOC
        level_match = re.search(r"(?:level|remaining capacity):\s*(\d+)", battery_dump, re.IGNORECASE)
        sys_soc = int(level_match.group(1)) if level_match else 0
        
        # 初始化 / 校准
        if last_real_cc is None and current_cc is not None:
            last_real_cc = current_cc
            if not is_calibrated and sys_soc > 0:
                calibrated_capacity_mah = (current_cc / 1000.0) / (sys_soc / 100.0)
                is_calibrated = True
                print(f"\n[校准] Init CC: {current_cc} uAh, Cap: {calibrated_capacity_mah:.2f} mAh")
        
        # 计算这一段周期的真实消耗
        if last_real_cc is not None and current_cc is not None:
            delta_cc = last_real_cc - current_cc # 放电时 last > current
            
            # 只有当消耗合理时才更新电流估算 (过滤掉 Reset 带来的充电干扰)
            # 如果 delta_cc < 0 (充进去了)，则忽略，保持之前的估算
            if delta_cc > 0:
                # Avg Current (mA) = (delta_uAh / 1000) / (Interval_s / 3600)
                interval = REFRESH_INTERVAL # 近似
                new_avg_current = (delta_cc / 1000.0) / (interval / 3600.0)
                # 平滑更新 (0.7 * old + 0.3 * new) 防止跳变
                estimated_current_ma = estimated_current_ma * 0.7 + new_avg_current * 0.3
                print(f" [修正] 真实消耗: {delta_cc} uAh, 新估算电流: {estimated_current_ma:.1f} mA")
            
            last_real_cc = current_cc
            
        # 记录电压 (仅刷新时更新)
        volt_match = re.search(r"voltage:\s*(\d+)", battery_dump, re.IGNORECASE)
        data["Voltage"] = int(volt_match.group(1)) if volt_match else 0

    # === 3. 生成输出数据 (无论是刷新还是盲区，都基于模型输出) ===
    
    # 死算 (Dead Reckoning) 当前的 CC
    # Current CC = Last Real CC - (Estimated Current * Time_Since_Last_Real)
    if last_real_cc is not None:
        elapsed = now - last_refresh_time
        consumed_uah = (estimated_current_ma * 1000.0) * (elapsed / 3600.0)
        current_cc_est = last_real_cc - consumed_uah
        
        # 计算 SOC
        cc_mah = current_cc_est / 1000.0
        if calibrated_capacity_mah > 0:
            soc = (cc_mah / calibrated_capacity_mah) * 100.0
            data["SOC"] = round(soc, 4)
        else:
            data["SOC"] = 0
    else:
        data["SOC"] = 0
        
    data["Current"] = int(estimated_current_ma)
    
    # 补全 Voltage (盲区时保持非零值，避免功率计算除零)
    if data["Voltage"] is None: data["Voltage"] = 4000 # 默认 4V

    return data

def get_cpu_info(battery_data):
    """获取CPU频率（GHz）、耗电功率（mW）"""
    cpu_data = {"CPU_Freq": 0.0, "CPU_Power": 0}
    
    # 轮询核心频率
    for cpu_idx in [7, 6, 4, 0]: # 优先大核
        path = f"/sys/devices/system/cpu/cpu{cpu_idx}/cpufreq/scaling_cur_freq"
        val = run_adb_cmd(f"cat {path}")
        if val.isdigit():
            cpu_data["CPU_Freq"] = round(int(val) / 1000000.0, 2)
            break
            
    # 功率估算
    if battery_data["Voltage"] and battery_data["Current"]:
        total_power = abs(battery_data["Voltage"] * battery_data["Current"]) / 1000.0
        # 异常值过滤 (>20W 视为单位错误)
        if total_power > 20000: total_power /= 1000.0
        
        # 负载比例
        max_freq = 3300000 # 默认 3.3GHz
        max_path = "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_max_freq"
        mf_val = run_adb_cmd(f"cat {max_path}")
        if mf_val.isdigit(): max_freq = int(mf_val)
        
        curr_khz = int(cpu_data["CPU_Freq"] * 1000000)
        load = curr_khz / max_freq if max_freq > 0 else 0
        cpu_data["CPU_Power"] = round(total_power * 0.6 * load, 2)
    
    return cpu_data

def get_network_info():
    """获取网络类型、RSRP、SINR、SNR"""
    net_data = {"Network_Type": 1, "RSRP": -999, "SINR": -999, "SNR": -999}
    
    wifi_status = run_adb_cmd("cmd wifi status")
    is_wifi = "Wifi is connected" in wifi_status or "mState=Connected" in run_adb_cmd("dumpsys wifi | grep mState")
    
    if is_wifi:
        net_data["Network_Type"] = 0
        # RSRP (RSSI)
        rssi = re.search(r"RSSI:\s*(-?\d+)", wifi_status, re.IGNORECASE)
        if rssi: net_data["RSRP"] = int(rssi.group(1))
        
        # SNR
        wifi_dump = run_adb_cmd("dumpsys wifi")
        snr = re.search(r"snr:\s*(-?\d+)", wifi_dump, re.IGNORECASE)
        if snr:
            net_data["SNR"] = int(snr.group(1))
        elif net_data["RSRP"] != -999:
            # 估算 SNR = RSSI - NoiseFloor(-95)
            net_data["SNR"] = net_data["RSRP"] - (-95)
    else:
        # Cellular
        telephony = run_adb_cmd("dumpsys telephony.registry")
        
        # RSRP
        rsrp = re.search(r"(?:rsrp|ssRsrp)(?:=|\s*:)\s*(-?\d+)", telephony, re.IGNORECASE) or \
               re.search(r"mRssi=(-?\d+)", telephony, re.IGNORECASE)
        if rsrp: net_data["RSRP"] = int(rsrp.group(1))
        
        # SINR / SNR
        sinr_pats = [r"(?:rssnr|ssSinr)(?:=|\s*:)\s*(-?\d+)", r"mSnr=(-?\d+)", r"sinr=(-?\d+)"]
        for pat in sinr_pats:
            m = re.search(pat, telephony, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if val < 1000: # 过滤非法大数
                    net_data["SINR"] = val
                    net_data["SNR"] = val # Cellular 下通常复用 SINR
                    break

    return net_data

def get_screen_brightness():
    val = run_adb_cmd("settings get system screen_brightness")
    return int(val) if val.isdigit() else 0

def get_background_app_count():
    ps = run_adb_cmd("ps -A")
    if "PID" not in ps: ps = run_adb_cmd("ps")
    count = 0
    for line in ps.split('\n'):
        if "u0_a" in line and "python" not in line and "adb" not in line:
            count += 1
    return count

def main():
    global OUTPUT_FILE
    
    # 确保电池服务状态正常
    print("正在初始化电池服务状态...")
    run_adb_cmd("dumpsys battery reset")
    
    headers = [
        "Time_Relative(s)", "Time_Str", "SOC(%)", "Battery_Temp(℃)",
        "CPU_Freq(GHz)", "CPU_Power(mW)", 
        "Screen_Brightness", "Network_Type", "RSRP(dBm)", "SINR(dB)", "SNR(dB)",
        "Ambient_Temp(℃)", "Background_App_Count"
    ]
    
    print(f"=== 电池数据采集系统 (脉冲更新修正版) ===")
    print(f"输出文件: {OUTPUT_FILE}")
    
    try:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers)
    except:
        OUTPUT_FILE = get_output_filename().replace(".csv", "_new.csv")
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers)
            
    print("开始采集... (按 Ctrl+C 停止)")
    start_time = time.time()
    
    try:
        while True:
            now = time.time()
            # 采集
            battery = get_battery_core_data()
            cpu = get_cpu_info(battery)
            net = get_network_info()
            
            row = [
                round(now - start_time, 2), datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                battery["SOC"], battery["Battery_Temp"],
                cpu["CPU_Freq"], cpu["CPU_Power"],
                get_screen_brightness(), net["Network_Type"], net["RSRP"], net["SINR"], net["SNR"],
                get_ambient_temp(), get_background_app_count()
            ]
            
            with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(row)
                
            print(f"\r[{row[0]}s] SOC:{battery['SOC']}% | Temp:{battery['Battery_Temp']} | SNR:{net['SNR']}", end="")
            time.sleep(SAMPLING_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n采集结束。请执行: adb shell dumpsys battery reset")

if __name__ == "__main__":
    main()
