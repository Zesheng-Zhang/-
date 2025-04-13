function [X_resampled, y_resampled] = smote_oversampling(X, y, k)
    if nargin < 3
        k = 5;
    end

    % 找出少数类样本
    minority_class = find(sum(y == 1) < sum(y == 0), 1, 'first');
    minority_indices = find(y == minority_class);
    X_minority = X(minority_indices, :);

    % 计算最近邻
    nbrs = knnsearch(X_minority, X_minority, 'K', k + 1);
    nbrs = nbrs(:, 2:end); % 排除自己

    % 生成新样本
    new_samples = [];
    for i = 1:size(X_minority, 1)
        nn = randi([1, k]);
        alpha = rand();
        new_sample = X_minority(i, :) + alpha * (X_minority(nbrs(i, nn), :) - X_minority(i, :));
        new_samples = [new_samples; new_sample];
    end

    % 打印原始数据和新样本的尺寸信息
    fprintf('原始特征矩阵 X 的尺寸: %d x %d\n', size(X, 1), size(X, 2));
    fprintf('新生成样本 new_samples 的尺寸: %d x %d\n', size(new_samples, 1), size(new_samples, 2));
    fprintf('原始标签向量 y 的长度: %d\n', length(y));

    % 合并原始数据和生成数据
    X_resampled = [X; new_samples];
    % 确保标签向量长度与特征矩阵一致
    y_resampled = [y; repmat(minority_class, size(new_samples, 1), 1)];

    % 打印合并后特征矩阵和标签向量的尺寸信息
    fprintf('合并后特征矩阵 X_resampled 的尺寸: %d x %d\n', size(X_resampled, 1), size(X_resampled, 2));
    fprintf('合并后标签向量 y_resampled 的长度: %d\n', length(y_resampled));

    % 检查合并过程中是否有异常
    if size(X, 1) + size(new_samples, 1) ~= size(X_resampled, 1)
        fprintf('合并特征矩阵时出现异常，预期行数: %d，实际行数: %d\n', size(X, 1) + size(new_samples, 1), size(X_resampled, 1));
    end
    if length(y) + size(new_samples, 1) ~= length(y_resampled)
        fprintf('合并标签向量时出现异常，预期长度: %d，实际长度: %d\n', length(y) + size(new_samples, 1), length(y_resampled));
    end

    % 再次检查长度是否一致
    if size(X_resampled, 1) ~= length(y_resampled)
        % 修正标签向量长度
        y_resampled = y_resampled(1:size(X_resampled, 1));
        if size(X_resampled, 1) ~= length(y_resampled)
            error('特征矩阵和标签向量长度不一致，请检查 SMOTE 过采样函数。');
        end
    end
end