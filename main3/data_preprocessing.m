
% 数据预处理函数
function [processed_data, vif_info] = data_preprocessing(data)
    % 1. Z-score标准化
    processed_data = zscore(data);
    
    % 2. 异常值处理(3σ原则)
    for col = 1:size(processed_data, 2)
        col_mean = mean(processed_data(:, col));
        col_std = std(processed_data(:, col));
        outliers = abs(processed_data(:, col) - col_mean) > 3 * col_std;
        processed_data(outliers, col) = col_mean;
    end
    
    % 3. 共线性检测(VIF)
    vif_info = zeros(size(processed_data, 2), 1);
    for i = 1:size(processed_data, 2)
        X = processed_data(:, [1:i-1, i+1:end]);
        y = processed_data(:, i);
        model = fitlm(X, y);
        r_squared = model.Rsquared.Ordinary;
        vif_info(i) = 1 / (1 - r_squared);
    end
    
    % 处理高共线性特征(VIF>10)
    high_vif_idx = find(vif_info > 10);
    if ~isempty(high_vif_idx)
        % 使用 sprintf 函数格式化字符串
        warning_str = sprintf('警告: 检测到高共线性特征 %s', num2str(high_vif_idx));
        disp(warning_str);
        processed_data(:, high_vif_idx) = [];
        vif_info(high_vif_idx) = [];
    end
end
