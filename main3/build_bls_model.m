
% 构建宽度学习系统(BLS)模型
function [Z_final, bls_params] = build_bls_model(X)
    n_feature_nodes = 10;
    n_enhance_nodes = 20;
    [n_samples, n_features] = size(X);

    % 1. 特征节点生成
    W_e = randn(n_features, n_feature_nodes);
    beta_e = randn(n_feature_nodes, 1);
    Z_e = sigmoid(X * W_e + repmat(beta_e', n_samples, 1));

    % 2. 增强节点生成
    Z_e_X = [Z_e, X];
    W_h = randn(size(Z_e_X, 2), n_enhance_nodes);
    Z_h = sigmoid(Z_e_X * W_h);

    % 3. 最终特征表示
    Z_final = [Z_e, Z_h];

    % 存储模型参数
    bls_params.W_e = W_e;
    bls_params.beta_e = beta_e;
    bls_params.W_h = W_h;
end

% 自定义sigmoid函数
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end
