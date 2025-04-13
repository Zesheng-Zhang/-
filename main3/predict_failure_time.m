
% 预测不合格事件发生时间
function predicted_times = predict_failure_time(cph, Z_final_test, time_horizon)
    risk_scores = Z_final_test * cph;
    predicted_times = zeros(size(Z_final_test, 1), 1);
    for i = 1:size(Z_final_test, 1)
        % 简单调整生存概率计算
        survival_prob = exp(-risk_scores(i)); 
        for t = 1:time_horizon
            if survival_prob < 0.5
                predicted_times(i) = t;
                break;
            end
            % 模拟生存概率随时间下降
            survival_prob = survival_prob * 0.95; 
        end
        if predicted_times(i) == 0
            predicted_times(i) = time_horizon;
        end
    end
end

