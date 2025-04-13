% 自定义函数5：Focal Loss
function loss = focal_loss(y_true, y_pred, gamma)
    if nargin < 3
        gamma = 2;
    end
    
    loss = - ((1 - y_pred).^gamma .* y_true .* log(y_pred) + ...
              y_pred.^gamma .* (1 - y_true) .* log(1 - y_pred));
    loss = mean(loss);
end
