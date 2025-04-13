% 特征工程函数
function X_fe = feature_engineering(X)
    [n_samples, n_features] = size(X);
    % 1. 生成交互项
    interaction_terms = [];
    for i = 1:n_features
        for j = i+1:n_features
            interaction_terms = [interaction_terms, X(:, i).* X(:, j)];
        end
    end
    
    % 2. 生成二次项
    quadratic_terms = X.^2;
    
    % 合并所有特征
    X_fe = [X, interaction_terms, quadratic_terms];
end