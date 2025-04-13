% 数据预处理函数
function [X_processed, y_processed, scaler_X, scaler_y, selected_features] = preprocess_data(X, y)
    % 1. Z-score标准化
    scaler_X.mean = mean(X);
    scaler_X.std = std(X);
    X_scaled = (X - repmat(scaler_X.mean, size(X, 1), 1))./ repmat(scaler_X.std, size(X, 1), 1);
    
    % 2. 异常值处理(3σ原则)
    for i = 1:size(X_scaled, 2)
        col_mean = mean(X_scaled(:, i));
        col_std = std(X_scaled(:, i));
        outliers = abs(X_scaled(:, i) - col_mean) > 3 * col_std;
        X_scaled(outliers, i) = col_mean;
    end
    
    % 3. 共线性检测(VIF)
    vif = zeros(1, size(X_scaled, 2));
    for i = 1:size(X_scaled, 2)
        X_temp = X_scaled(:, [1:i-1, i+1:end]);
        y_temp = X_scaled(:, i);
        model = fitlm(X_temp, y_temp);
        r_squared = model.Rsquared.Ordinary;
        vif(i) = 1 / (1 - r_squared);
    end
    
    % 移除高VIF的特征(VIF>10)
    selected_features = find(vif <= 10);
    X_processed = X_scaled(:, selected_features);
    
    % 输出数据标准化
    scaler_y.mean = mean(y);
    scaler_y.std = std(y);
    y_processed = (y - repmat(scaler_y.mean, size(y, 1), 1))./ repmat(scaler_y.std, size(y, 1), 1);
end
