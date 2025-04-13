
% 时间序列特征工程
function enhanced_data = time_series_feature_engineering(data)
    window_size = 5;
    lag_steps = 5;

    % 预分配内存以提高性能
    num_cols = size(data, 2);
    new_cols = num_cols * (3 + lag_steps);
    enhanced_data = zeros(size(data, 1), size(data, 2) + new_cols);
    enhanced_data(:, 1:num_cols) = data;

    % 1. 滑动窗口统计量
    for col = 1:num_cols
        rolling_mean = movmean(data(:, col), window_size);
        rolling_std = movstd(data(:, col), window_size);
        rolling_max = movmax(data(:, col), window_size);
        start_idx = num_cols + (col - 1) * 3 + 1;
        enhanced_data(:, start_idx:start_idx + 2) = [rolling_mean, rolling_std, rolling_max];
    end

    % 2. 滞后变量构建
    for col = 1:num_cols
        for lag = 1:lag_steps
            lagged_col = [nan(lag, 1); data(1:end - lag, col)];
            start_idx = num_cols + num_cols * 3 + (col - 1) * lag_steps + lag;
            enhanced_data(:, start_idx) = lagged_col;
        end
    end

    % 删除因滑动窗口和滞后产生的NaN值
    enhanced_data = rmmissing(enhanced_data);
end
