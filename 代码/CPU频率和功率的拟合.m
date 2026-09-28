% 清空工作区和命令行
clear; clc; close all;

% 定义文件名
filename = '打游戏_正常亮度.csv';

% 检查文件是否存在
if ~isfile(filename)
    error('错误：文件 "%s" 未在当前目录下找到。请确认文件路径。', filename);
end

% 读取CSV文件
% 使用 detectImportOptions 并设置 VariableNamingRule 为 preserve 以保留原始列名
opts = detectImportOptions(filename);
opts.VariableNamingRule = 'preserve';
data = readtable(filename, opts);

% 提取数据
% 根据文件内容：第5列是 CPU_Freq(GHz)，第6列是 CPU_Power(mW)
% 这里直接使用列索引提取，避免列名中特殊字符的问题
try
    cpu_freq = data{:, 5}; % x: CPU频率
    cpu_power = data{:, 6}./1000; % y: CPU功率
catch
    error('错误：读取数据列失败，请检查CSV文件格式是否正确（第5列应为频率，第6列应为功率）。');
end

% 数据预处理：去除可能存在的 NaN 值
valid_indices = ~isnan(cpu_freq) & ~isnan(cpu_power);
cpu_freq = cpu_freq(valid_indices);
cpu_power = cpu_power(valid_indices);

% 线性拟合
% 目标模型：CPU_Power = P_0 + k_0 * CPU_Freq
% polyfit(x, y, 1) 返回 [斜率, 截距]，即 [k_0, P_0]
p = polyfit(cpu_freq, cpu_power, 1);
k_0 = p(1);
P_0 = p(2);

% --- 计算 R^2 (决定系数) ---
y_pred = polyval(p, cpu_freq);             % 计算拟合值
y_mean = mean(cpu_power);                  % 计算均值
SS_tot = sum((cpu_power - y_mean).^2);     % 总离差平方和
SS_res = sum((cpu_power - y_pred).^2);     % 残差平方和
R_squared = 1 - (SS_res / SS_tot);         % R^2 计算公式

% 计算拟合直线上的点用于绘图
x_fit = linspace(min(cpu_freq), max(cpu_freq), 100);
y_fit = polyval(p, x_fit);

% 绘图结果可视化
figure('Name', 'CPU功率与频率线性拟合', 'Color', 'w');
scatter(cpu_freq, cpu_power, 50, 'b', 'filled', 'MarkerFaceAlpha', 0.5); % 原始数据散点
hold on;
plot(x_fit, y_fit, 'r-', 'LineWidth', 2); % 拟合直线
grid on;

% 添加图表标注
xlabel('CPU Frequency (GHz)');
ylabel('CPU Power (W)');
% 标题中加入 R^2 信息
title({['CPU Power vs Frequency Fitting'], ...
       ['P = ' num2str(P_0, '%.2f') ' + ' num2str(k_0, '%.2f') ' * f,  R^2 = ' num2str(R_squared, '%.4f')]});
legend('Original Data', 'Linear Fit', 'Location', 'NorthWest');

% 在控制台输出结果
disp('--------------------------------------------------');
disp('拟合结果 (Fitting Results):');
disp(['P_0 (截距) = ', num2str(P_0)]);
disp(['k_0 (斜率) = ', num2str(k_0)]);
disp(['R^2 (决定系数) = ', num2str(R_squared)]);
disp('--------------------------------------------------');
disp(['拟合方程: CPU_Power = ', num2str(P_0), ' + ', num2str(k_0), ' * CPU_Freq']);
