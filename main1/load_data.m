
% 加载数据函数
function [X, y] = load_data(input_file, output_file)
    % 读取输入数据，假设输入文件没有表头
    X = readmatrix(input_file);
    % 读取输出数据，假设输出文件第一行是表头
    y = readmatrix(output_file, 'HeaderLines', 1);
end