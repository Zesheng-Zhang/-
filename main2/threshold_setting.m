function [k1, k2] = threshold_setting(output_data)
    % 将 output_data 从 table 类型转换为数值矩阵
    output_matrix = table2array(output_data);
    
    % 计算二氧化硫阈值
    k1_percentile = prctile(output_matrix(:, 1), 95);
    k1 = k1_percentile;
    
    % 计算硫化氢阈值
    k2_percentile = prctile(output_matrix(:, 2), 95);
    k2 = k2_percentile;
end