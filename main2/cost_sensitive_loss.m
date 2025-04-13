% 自定义函数4：代价敏感学习
function loss = cost_sensitive_loss(y_true, y_pred, w_tp, w_fp, w_fn)
    if nargin < 4
        w_tp = 1;
    end
    if nargin < 5
        w_fp = 2;
    end
    if nargin < 6
        w_fn = 1;
    end
    
    loss = - (w_tp * y_true .* log(y_pred) + w_fp * (1 - y_true) .* log(1 - y_pred) + ...
              w_fn * (1 - y_true) .* log(y_pred) + w_tp * y_true .* log(1 - y_pred));
    loss = mean(loss);
end


