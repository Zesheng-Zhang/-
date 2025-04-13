% 可视化结果
function visualize_results(Z_final, event_times, event_observed, predicted_times, cph, enhanced_data, bls_params, test_idx, time_horizon)
    % 绘制事件发生时间分布
    figure;
    histogram(event_times, 'Normalization', 'probability');
    title('事件发生时间分布');
    xlabel('事件发生时间');
    ylabel('概率');

    % 绘制预测结果与实际事件时间对比（前20个样本）
    figure;
    plot(1:min(20, length(predicted_times)), predicted_times(1:min(20, length(predicted_times))), 'ro', 'DisplayName', '预测时间');
    hold on;
    plot(1:min(20, length(event_times)), event_times(1:min(20, length(event_times))), 'bo', 'DisplayName', '实际时间');
    title('预测结果与实际事件时间对比（前20个样本）');
    xlabel('样本编号');
    ylabel('时间');
    legend;

    % 绘制生存函数曲线（前5个样本）
    figure;
    risk_scores_5 = Z_final(1:5, :) * cph;
    for i = 1:5
        survival_prob = exp(-risk_scores_5(i));
        survival_probs = zeros(time_horizon, 1);
        for t = 1:time_horizon
            survival_probs(t) = survival_prob;
            survival_prob = survival_prob * 0.95;
        end
        plot(1:time_horizon, survival_probs, 'DisplayName', ['样本', num2str(i)]);
        hold on;
    end
    title('前5个样本的生存函数曲线');
    xlabel('时间');
    ylabel('生存概率');
    legend;

    % 绘制特征分布（前5个特征）
    figure;
    for i = 1:min(5, size(enhanced_data, 2))
        subplot(5, 1, i);
        histogram(enhanced_data(:, i), 'Normalization', 'probability');
        title(['特征 ', num2str(i), ' 分布']);
        xlabel('特征值');
        ylabel('概率');
    end

    % 绘制特征相关性矩阵
    figure;
    corr_matrix = corr(enhanced_data);
    imagesc(corr_matrix);
    colorbar;
    title('特征相关性矩阵');
    xlabel('特征编号');
    ylabel('特征编号');

    % 绘制Cox模型系数
    figure;
    bar(cph);
    title('Cox模型系数');
    xlabel('特征编号');
    ylabel('系数值');

    % 绘制预测误差分布
    figure;
    error = abs(predicted_times - event_times(test_idx));
    histogram(error, 'Normalization', 'probability');
    title('预测误差分布');
    xlabel('预测误差');
    ylabel('概率');

    % 绘制BLS特征节点权重矩阵热力图
    figure;
    imagesc(bls_params.W_e);
    colorbar;
    title('BLS特征节点权重矩阵热力图');
    xlabel('特征节点编号');
    ylabel('输入特征编号');

    % 绘制BLS增强节点权重矩阵热力图
    figure;
    imagesc(bls_params.W_h);
    colorbar;
    title('BLS增强节点权重矩阵热力图');
    xlabel('增强节点编号');
    ylabel('合并特征编号');

    % 绘制特征节点输出分布（前5个特征节点）
    Z_e = sigmoid(enhanced_data * bls_params.W_e + repmat(bls_params.beta_e', size(enhanced_data, 1), 1));
    figure;
    for i = 1:min(5, size(Z_e, 2))
        subplot(5, 1, i);
        histogram(Z_e(:, i), 'Normalization', 'probability');
        title(['特征节点 ', num2str(i), ' 输出分布']);
        xlabel('特征节点输出值');
        ylabel('概率');
    end

    % 绘制增强节点输出分布（前5个增强节点）
    Z_e_X = [Z_e, enhanced_data];
    Z_h = sigmoid(Z_e_X * bls_params.W_h);
    figure;
    for i = 1:min(5, size(Z_h, 2))
        subplot(5, 1, i);
        histogram(Z_h(:, i), 'Normalization', 'probability');
        title(['增强节点 ', num2str(i), ' 输出分布']);
        xlabel('增强节点输出值');
        ylabel('概率');
    end

    % 绘制预测时间与实际时间的散点图
    figure;
    scatter(event_times(test_idx), predicted_times);
    xlabel('实际事件时间');
    ylabel('预测事件时间');
    title('预测时间与实际时间的散点图');
    line([min(event_times(test_idx)), max(event_times(test_idx))], [min(event_times(test_idx)), max(event_times(test_idx))], 'Color', 'r');

    % 绘制不同生存状态下特征均值对比（前5个特征）
    censored_idx = event_observed == 0;
    uncensored_idx = event_observed == 1;
    figure;
    for i = 1:min(5, size(enhanced_data, 2))
        subplot(5, 1, i);
        bar([mean(enhanced_data(censored_idx, i)), mean(enhanced_data(uncensored_idx, i))]);
        title(['特征 ', num2str(i), ' 在不同生存状态下的均值对比']);
        xlabel('生存状态（1: 删失，2: 事件发生）');
        ylabel('特征均值');
    end
end