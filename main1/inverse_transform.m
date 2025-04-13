% 反标准化函数
function y_orig = inverse_transform(y, scaler)
    y_orig = y .* repmat(scaler.std, size(y, 1), 1) + repmat(scaler.mean, size(y, 1), 1);
end