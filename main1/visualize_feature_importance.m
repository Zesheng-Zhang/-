% 可视化特征重要性
function visualize_feature_importance(beta, X_fe)
    % 计算特征重要性（简单地使用输出层权重的绝对值）
    feature_importance = sum(abs(beta), 2);
    
    % 绘制特征重要性柱状图
    figure;
    bar(feature_importance);
    title('特征重要性');
    xlabel('特征编号');
    ylabel('重要性');
end