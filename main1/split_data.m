% 划分数据集函数
function [X_train, X_test, y_train, y_test] = split_data(X, y, test_size, random_state)
    rng(random_state);
    indices = randperm(size(X, 1));
    test_indices = indices(1:round(test_size * size(X, 1)));
    train_indices = indices(round(test_size * size(X, 1))+1:end);
    
    X_train = X(train_indices, :);
    X_test = X(test_indices, :);
    y_train = y(train_indices, :);
    y_test = y(test_indices, :);
end