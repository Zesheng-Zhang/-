% 主函数
function fitness=main(M,N)
% clc;clear;
    % 1. 加载数据
    % 请将以下文件路径替换为实际的数据文件路径
    input_file = 'IN_Table.csv';
    output_file = 'OUT_Table.csv';
   [X, y] = load_data(input_file, output_file);
    X=X(1:200,:);
    y=y(1:200,:);
    % 2. 数据预处理
    [X_processed, y_processed, scaler_X, scaler_y, selected_features] = preprocess_data(X, y);
    
    % 3. 特征工程
    X_fe = feature_engineering(X_processed);
    
    % 4. 划分训练集和测试集
    [X_train, X_test, y_train, y_test] = split_data(X_fe, y_processed, 0.3, 42);
    
    % 5. 初始化并训练BLS模型
    n_feature_nodes = M;
    n_enhance_nodes = N;
    bls = BLS(n_feature_nodes, n_enhance_nodes);
    bls = bls.fit(X_train, y_train);
    
    % 6. 预测并评估模型
    y_train_pred = bls.predict(X_train);
    y_test_pred = bls.predict(X_test);
    
    % 反标准化预测结果
    y_train_pred_orig = inverse_transform(y_train_pred, scaler_y);
    y_test_pred_orig = inverse_transform(y_test_pred, scaler_y);
    y_train_orig = inverse_transform(y_train, scaler_y);
    y_test_orig = inverse_transform(y_test, scaler_y);
    
    % 计算评估指标
    [train_mse, train_r2] = evaluate_model(y_train_orig, y_train_pred_orig);
    [test_mse, test_r2] = evaluate_model(y_test_orig, y_test_pred_orig);
    
    fprintf('训练集性能:\n');
    fprintf('均方误差(MSE): %.4f\n', train_mse);
    fprintf('决定系数(R²): %.4f\n', train_r2);
    
    fprintf('\n测试集性能:\n');
    fprintf('均方误差(MSE): %.4f\n', test_mse);
    fprintf('决定系数(R²): %.4f\n', test_r2);
    fitness=test_mse;
    % % 
    % % 可视化训练集和测试集的真实值与预测值
    % visualize_results(y_train_orig, y_train_pred_orig, y_test_orig, y_test_pred_orig);
    % 
    % % 可视化特征重要性
    % visualize_feature_importance(bls.beta, X_fe);
    % 
    % % 可视化训练集和测试集的残差图
    % visualize_residuals(y_train_orig, y_train_pred_orig, y_test_orig, y_test_pred_orig);
end