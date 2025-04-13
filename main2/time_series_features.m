function X_features = time_series_features(input_data, output_data, window_size, lag_size)
    if nargin < 3
        window_size = 5;
    end
    if nargin < 4
        lag_size = 5;
    end
    
    n_samples = size(input_data, 1);
    n_features = size(input_data, 2);
    new_features = [];
    
    for i = window_size:n_samples
        % 输入数据的窗口统计
        window_input = input_data(i-window_size+1:i, :);
        input_mean = mean(window_input);
        input_std = std(window_input);
        input_max = max(window_input);
        
        % 输出数据的窗口统计
        window_output = output_data(i-window_size+1:i, :);
        output_mean = mean(window_output);
        output_std = std(window_output);
        output_max = max(window_output);
        
        % 滞后变量
        lag_input = zeros(lag_size, n_features);
        lag_output = zeros(lag_size, 2);
        for l = 1:lag_size
            if i-l >= 1
                lag_input(l, :) = input_data(i-l, :);
                lag_output(l, :) = output_data(i-l, :);
            end
        end
        
        % 确保所有数组为行向量
        input_data_row = input_data(i, :);
        output_data_row = output_data(i, :);
        input_mean_row = input_mean(:)';
        input_std_row = input_std(:)';
        input_max_row = input_max(:)';
        output_mean_row = output_mean(:)';
        output_std_row = output_std(:)';
        output_max_row = output_max(:)';
        lag_input_row = lag_input(:)';
        lag_output_row = lag_output(:)';
        
        % 合并所有特征
        current_features = [input_data_row, output_data_row, input_mean_row, input_std_row, input_max_row, output_mean_row, output_std_row, output_max_row, lag_input_row, lag_output_row];
        new_features = [new_features; current_features];
    end
    
    X_features = new_features;
end