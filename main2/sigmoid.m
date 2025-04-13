% 自定义函数9：Sigmoid激活函数
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end