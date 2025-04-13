% 可视化真实值与预测值
function visualize_results(y_train_orig, y_train_pred_orig, y_test_orig, y_test_pred_orig)
    % 绘制训练集真实值与预测值对比图（第一个输出变量）
    figure;
    subplot(2,2,1);
    hold on;
    plot(y_train_orig(:,1), 'b-o', 'DisplayName', '真实值');
    plot(y_train_pred_orig(:,1), 'r-s', 'DisplayName', '预测值');
    title('训练集真实值与预测值对比（第一个输出变量）');
    xlabel('样本编号');
    ylabel('值');
    legend;
    
    % 绘制训练集真实值与预测值对比图（第二个输出变量）
    subplot(2,2,2);
    hold on;
    plot(y_train_orig(:,2), 'b-o', 'DisplayName', '真实值');
    plot(y_train_pred_orig(:,2), 'r-s', 'DisplayName', '预测值');
    title('训练集真实值与预测值对比（第二个输出变量）');
    xlabel('样本编号');
    ylabel('值');
    legend;
    
    % 绘制测试集真实值与预测值对比图（第一个输出变量）
    subplot(2,2,3);
    hold on;
    plot(y_test_orig(:,1), 'b-o', 'DisplayName', '真实值');
    plot(y_test_pred_orig(:,1), 'r-s', 'DisplayName', '预测值');
    title('测试集真实值与预测值对比（第一个输出变量）');
    xlabel('样本编号');
    ylabel('值');
    legend;
    
    % 绘制测试集真实值与预测值对比图（第二个输出变量）
    subplot(2,2,4);
    hold on;
    plot(y_test_orig(:,2), 'b-o', 'DisplayName', '真实值');
    plot(y_test_pred_orig(:,2), 'r-s', 'DisplayName', '预测值');
    title('测试集真实值与预测值对比（第二个输出变量）');
    xlabel('样本编号');
    ylabel('值');
    legend;
end
