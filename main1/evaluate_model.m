% 评估模型函数
function [mse, r2] = evaluate_model(y_true, y_pred)
    mse = mean((y_true - y_pred).^2, 'all');
    ss_res = sum((y_true - y_pred).^2, 'all');
    ss_tot = sum((y_true - mean(y_true, 'all')).^2, 'all');
    r2 = 1 - ss_res / ss_tot;
end