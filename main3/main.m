% 主程序
function main()
    % 清空命令窗口和工作区，设置随机数种子以保证结果可复现
    clc; clear;
    rng(42); 

    % 定义输入文件路径
    input_file = 'IN_Table.csv';
    output_file = 'OUT_Table.csv';

    % 读取文件数据
    try
        input_data = readtable(input_file);
        output_data = readtable(output_file);
    catch ME
        fprintf('读取文件时出错: %s\n', ME.message);
        return;
    end

    % 去除第一行数据
    input_data = input_data(2:end, :);
    output_data = output_data(2:end, :);

    % 合并数据
    data = [input_data, output_data];

    % 转换为数值矩阵
    data = table2array(data);

    % 数据预处理
    [processed_data, vif_info] = data_preprocessing(data);
    disp('共线性检测结果:');
    disp(vif_info);

    % 时间序列特征工程
    enhanced_data = time_series_feature_engineering(processed_data);

    % 构建BLS模型
    [Z_final, bls_params] = build_bls_model(enhanced_data);

    % 准备生存分析数据
    % 在实际应用中，这部分数据应从问题二的不合格样本中获取
    event_times = randi([10, 70], size(Z_final, 1), 1); % 生成模拟的事件发生时间
    event_observed = binornd(1, 0.8, size(Z_final, 1), 1); % 生成模拟的事件观察状态

    % 生存分析模型训练
    [cph, ~, ~, ~] = survival_analysis(Z_final, event_times, event_observed);

    % 模型预测 (使用部分数据作为测试集)
    test_idx = randi(size(Z_final, 1), 50, 1); % 随机选择50个样本作为测试集
    Z_final_test = Z_final(test_idx, :);
    time_horizon = 70; % 定义时间范围
    predicted_times = predict_failure_time(cph, Z_final_test, time_horizon);

    % 打印前10个样本的预测结果
    disp('前10个样本的预测结果:');
    for i = 1:min(10, length(predicted_times))
        fprintf('样本%d: 预测不合格时间 = %d\n', i, predicted_times(i));
    end

    % 可视化部分结果
    visualize_results(Z_final, event_times, event_observed, predicted_times, cph, enhanced_data, bls_params, test_idx, time_horizon);
end
