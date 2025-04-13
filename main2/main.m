% 主函数开始
function main()
    clc;clear;
    rng(42); % 设置随机数种子以保证结果可复现
    input_file = 'IN_Table.csv';
    output_file = 'OUT_Table.csv';
    input_data = readtable(input_file);
    output_data = readtable(output_file);
    % 去除第一行数据
    input_data = input_data(2:end, :);
    output_data = output_data(2:end, :);

    % 2. 数据预处理
    [input_processed, output_processed, kept_cols] = data_preprocessing(input_data, output_data);
    fprintf('原始输入特征数: %d, 处理后特征数: %d\n', size(input_data, 2), length(kept_cols));

    % 可视化原始输入数据的部分特征
    visualize_input_features(input_data, input_processed);

    % 3. 阈值设定
    [k1, k2] = threshold_setting(output_data);
    fprintf('二氧化硫阈值: %.4f, 硫化氢阈值: %.4f\n', k1, k2);

    % 4. 创建标签 (根据阈值判断是否合格)
    y = (output_processed(:, 1) > k1 | output_processed(:, 2) > k2);
    fprintf('类别分布: 合格 %d, 不合格 %d\n', sum(~y), sum(y));

    % 可视化输出数据及阈值
    visualize_output_threshold(output_processed, k1, k2);

    % 5. 时间序列特征工程
    X_features = time_series_features(input_processed, output_processed);
    fprintf('时间序列特征工程后特征维度: %d\n', size(X_features, 2));

    % 可视化部分时间序列特征
    visualize_time_series_features(X_features);

    % 6. 处理不平衡数据 (SMOTE过采样)
    [X_resampled, y_resampled] = smote_oversampling(X_features, y);
    fprintf('过采样后类别分布: 合格 %d, 不合格 %d\n', sum(~y_resampled), sum(y_resampled));

    % 检查 X_resampled 和 y_resampled 的长度是否一致
    check_resampled_length(X_resampled, y_resampled);

    % 7. 划分训练集和测试集
    [X_train, y_train, X_test, y_test] = split_train_test(X_resampled, y_resampled);

    % 8. 训练宽度学习系统
    bls = BroadLearningSystem(20, 10);
    bls = bls.fit(X_train, y_train, 0.01, 100, 32, 'focal');

    % 9. 评估模型
    [y_pred] = evaluate_model(bls, X_test, y_test);

    % 10. 可视化部分结果
    visualize_results(X_test, y_test, y_pred);
end

% 可视化原始输入数据的部分特征
function visualize_input_features(input_data, input_processed)
    % 检查 input_data 是否为 table 类型，若是则转换为数值矩阵
    if isa(input_data, 'table')
        input_data_matrix = table2array(input_data);
    else
        input_data_matrix = input_data;
    end

    % 检查 input_processed 是否为 table 类型，若是则转换为数值矩阵
    if isa(input_processed, 'table')
        input_processed_matrix = table2array(input_processed);
    else
        input_processed_matrix = input_processed;
    end

    figure;
    subplot(2, 1, 1);
    plot(input_data_matrix(:, 1:2));
    title('原始输入数据部分特征');
    xlabel('样本编号');
    ylabel('特征值');
    legend('特征1', '特征2');

    subplot(2, 1, 2);
    plot(input_processed_matrix(:, 1:2));
    title('处理后输入数据部分特征');
    xlabel('样本编号');
    ylabel('特征值');
    legend('特征1', '特征2');
end

% 可视化输出数据及阈值
function visualize_output_threshold(output_processed, k1, k2)
    n_samples = size(output_processed, 1);
    figure;
    subplot(2, 1, 1);
    plot(output_processed(:, 1));
    hold on;
    plot([1 n_samples], [k1 k1], 'r--');
    title('二氧化硫浓度及阈值');
    xlabel('样本编号');
    ylabel('浓度');
    legend('二氧化硫浓度', '阈值');

    subplot(2, 1, 2);
    plot(output_processed(:, 2));
    hold on;
    plot([1 n_samples], [k2 k2], 'r--');
    title('硫化氢浓度及阈值');
    xlabel('样本编号');
    ylabel('浓度');
    legend('硫化氢浓度', '阈值');
end

% 可视化部分时间序列特征
function visualize_time_series_features(X_features)
    figure;
    plot(X_features(:, 1:2));
    title('部分时间序列特征');
    xlabel('样本编号');
    ylabel('特征值');
    legend('特征1', '特征2');
end

% 检查过采样后特征矩阵和标签向量长度是否一致
function check_resampled_length(X_resampled, y_resampled)
    if length(X_resampled) ~= length(y_resampled)
        error('特征矩阵和标签向量长度不一致，请检查 SMOTE 过采样函数。');
    end
end

% 划分训练集和测试集
function [X_train, y_train, X_test, y_test] = split_train_test(X_resampled, y_resampled)
    cv = cvpartition(y_resampled, 'HoldOut', 0.2);
    idxTrain = training(cv);
    idxTest = test(cv);

    % 确保索引在有效范围内
    if max(idxTrain) > size(X_resampled, 1) || max(idxTest) > size(X_resampled, 1)
        error('生成的索引超出特征矩阵的范围，请检查数据划分过程。');
    end

    X_train = X_resampled(idxTrain, :);
    y_train = y_resampled(idxTrain);
    X_test = X_resampled(idxTest, :);
    y_test = y_resampled(idxTest);
end

% 评估模型
function [y_pred] = evaluate_model(bls, X_test, y_test)
    y_pred = bls.predict(X_test);

    % 将 y_test 和 y_pred 转换为相同的数据类型（这里转换为双精度类型）
    y_test = double(y_test);
    y_pred = double(y_pred);

    confMat = confusionmat(y_test, y_pred);
    classReport = classification_report(y_test, y_pred);
    disp('分类报告:');
    disp(classReport);
    disp('混淆矩阵:');
    disp(confMat);

    % 可视化混淆矩阵
    visualize_confusion_matrix(confMat);
end

% 可视化混淆矩阵
function visualize_confusion_matrix(confMat)
    figure;
    imagesc(confMat);
    colorbar;
    title('混淆矩阵');
    xlabel('预测类别');
    ylabel('真实类别');
    set(gca, 'XTick', [1 2], 'XTickLabel', {'合格', '不合格'});
    set(gca, 'YTick', [1 2], 'YTickLabel', {'合格', '不合格'});
    for i = 1:size(confMat, 1)
        for j = 1:size(confMat, 2)
            text(j, i, num2str(confMat(i, j)), 'HorizontalAlignment', 'center');
        end
    end
end

% 可视化部分结果
function visualize_results(X_test, y_test, y_pred)
    figure;
    subplot(1, 2, 1);
    scatter(X_test(:, 1), X_test(:, 2), 20, y_test, 'filled');
    title('真实标签');
    xlabel('特征1');
    ylabel('特征2');
    % 手动定义类似 coolwarm 的颜色映射
    coolwarm_map = create_coolwarm_map(64);
    colormap(coolwarm_map);
    colorbar;

    subplot(1, 2, 2);
    scatter(X_test(:, 1), X_test(:, 2), 20, y_pred, 'filled');
    title('预测标签');
    xlabel('特征1');
    ylabel('特征2');
    % 手动定义 coolwarm 颜色映射
    coolwarm_map = create_coolwarm_map(256);
    colormap(coolwarm_map);
    colorbar;
end

% 创建 coolwarm 颜色映射
function coolwarm_map = create_coolwarm_map(n)
    coolwarm_map = zeros(n, 3);
    for i = 1:n
        t = (i - 1) / (n - 1);
        coolwarm_map(i, 1) = 0.2372 + 2.1256*t - 1.6036*t^2;
        coolwarm_map(i, 2) = 0.2372 + 2.1256*t - 1.6036*t^2;
        coolwarm_map(i, 3) = 0.8627 - 1.2274*t + 0.3647*t^2;
    end
end