
% 生存分析(Cox比例风险模型)
function [cph, beta, logL, stats] = survival_analysis(Z_final, event_times, event_observed)
    % 准备生存分析数据，将起始时间设为 0
    start_times = zeros(size(event_times));
    % 构建两列的时间数据矩阵
    time_data = [start_times, event_times];
    % 使用 coxphfit 函数拟合 Cox 比例风险模型
    try
        [beta, logL, stats] = coxphfit(Z_final, time_data, 'Censoring', event_observed);
    catch ME
        fprintf('生存分析模型训练时出错: %s\n', ME.message);
        cph = [];
        beta = [];
        logL = [];
        stats = [];
        return;
    end
    % 这里为了保持与原代码结构一致，将 beta 作为 cph 返回，实际使用时可按需调整
    cph = beta;
end
