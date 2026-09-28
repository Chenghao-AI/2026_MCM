% 清空工作区
clear; clc; close all;

% --- 1. 读取数据 ---
filename = 'data.xlsx';
if ~isfile(filename)
    % 如果文件不存在，手动创建示例数据（基于用户描述，防止运行报错）
    disp('未找到 data.xlsx，正在生成示例数据用于演示...');
    T = [-40; -20; -10; 0; 25; 40; 55; 60];
    V = [4865.3; 7800.6; 9192.4; 10181.4; 10432.4; 11572; 10899.7; 10340.9];
    % 保存为临时文件以便后续统一处理
    writetable(table(T, V, 'VariableNames', {'Temperature/(◦C)', 'Mean Value'}), filename);
end

% 读取数据
opts = detectImportOptions(filename);
opts.VariableNamingRule = 'preserve';
data = readtable(filename, opts);

% 提取变量
% 假设第一列是温度，第二列是均值
x = data{:, 1}; 
y = data{:, 2};

% 去除NaN
valid = ~isnan(x) & ~isnan(y);
x = x(valid);
y = y(valid);

% --- 2. 拟合模型：三次多项式 (Cubic Polynomial) ---
% 拟合模型：y = p1*x^3 + p2*x^2 + p3*x + p4
n = 3; % 三次多项式
p = polyfit(x, y, n);

% 计算拟合值
y_fit = polyval(p, x);
sse = sum((y - y_fit).^2); % 残差平方和

% 生成显示用的公式字符串
str = 'y = ';
for i = 1:n+1
    coeff = p(i);
    power = n - i + 1;
    
    % 处理正负号显示
    if i > 1
        if coeff >= 0
            str = [str, '+ '];
        else
            str = [str, '- '];
            coeff = abs(coeff); % 后面直接显示绝对值
        end
    else
        % 第一项如果是负数
        if coeff < 0
            str = [str, '-'];
            coeff = abs(coeff);
        end
    end
    
    if power > 1
        str = [str, sprintf('%.4f*x^%d ', coeff, power)];
    elseif power == 1
        str = [str, sprintf('%.4f*x ', coeff)];
    else
        str = [str, sprintf('%.4f', coeff)];
    end
end

% --- 3. 结果可视化 ---
figure('Name', '三次函数拟合结果', 'Color', 'w');
scatter(x, y, 60, 'k', 'filled', 'o'); hold on;

% 生成平滑曲线用于绘图
x_smooth = linspace(min(x), max(x), 200);
y_smooth = polyval(p, x_smooth);

plot(x_smooth, y_smooth, 'r-', 'LineWidth', 2);
grid on;
legend('原始数据', '三次拟合曲线', 'Location', 'best');
xlabel('Temperature (^{\circ}C)');
ylabel('Mean Value');
title('三次函数拟合 (Cubic Fit)'); % 标题简化，信息移至图中

% 在图像中添加文本框显示公式和SSE
% 设置文本框位置（根据数据范围动态调整）
x_range = max(x) - min(x);
y_range = max(y) - min(y);
text_x = min(x) + 0.05 * x_range; % 左侧留白5%
text_y = max(y) - 0.05 * y_range; % 顶部留白5%

% 构建文本内容
info_str = {['Fit: ' str], ['SSE: ' num2str(sse, '%.2f')]};

% 显示文本框
text(text_x, text_y, info_str, 'VerticalAlignment', 'top', ...
    'FontSize', 10, 'BackgroundColor', 'w', 'EdgeColor', 'k', 'Interpreter', 'none');

% --- 4. 输出结果 ---
disp('==================================================');
disp('                三次多项式拟合结果                ');
disp('==================================================');
disp(['拟合函数表达式: ', str]);
disp(['SSE (残差平方和): ', num2str(sse)]);
disp('--------------------------------------------------');
disp('系数向量 (从高次到低次):');
disp(p);
disp('==================================================');
