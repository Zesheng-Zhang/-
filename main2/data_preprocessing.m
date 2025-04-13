function [input_processed, output_processed, kept_cols] = data_preprocessing(input_data, output_data)
    % 将输入数据从 table 类型转换为数值矩阵
    input_matrix = table2array(input_data);
    % 对输入数据进行标准化处理
    input_scaled = zscore(input_matrix);
    
    % 这里假设对输出数据也进行类似处理，同样先转换为矩阵
    output_matrix = table2array(output_data);
    output_scaled = zscore(output_matrix);
    
    % 假设这里有一些特征选择的逻辑，例如保留某些列
    kept_cols = 1:size(input_scaled, 2); % 这里简单示例保留所有列
    
    input_processed = input_scaled;
    output_processed = output_scaled;
end