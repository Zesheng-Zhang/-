
% 可视化残差图
function visualize_residuals(y_train_orig, y_train_pred_orig, y_test_orig, y_test_pred_orig)
    % 计算训练集和测试集的残差
    train_residuals = y_train_orig - y_train_pred_orig;
    test_residuals = y_test_orig - y_test_pred_orig;
    
    % 绘制训练集残差图（第一个输出变量）
    figure;
    subplot(2,2,1);
    scatter(y_train_pred_orig(:,1), train_residuals(:,1));
    title('训练集残差图（第一个输出变量）');
    xlabel('预测值');
    ylabel('残差');
    grid on;
    
    % 绘制训练集残差图（第二个输出变量）
    subplot(2,2,2);
    scatter(y_train_pred_orig(:,2), train_residuals(:,2));
    title('训练集残差图（第二个输出变量）');
    xlabel('预测值');
    ylabel('残差');
    grid on;
    
    % 绘制测试集残差图（第一个输出变量）
    subplot(2,2,3);
    scatter(y_test_pred_orig(:,1), test_residuals(:,1));
    title('测试集残差图（第一个输出变量）');
    xlabel('预测值');
    ylabel('残差');
    grid on;
    
    % 绘制测试集残差图（第二个输出变量）
    subplot(2,2,4);
    scatter(y_test_pred_orig(:,2), test_residuals(:,2));
    title('测试集残差图（第二个输出变量）');
    xlabel('预测值');
    ylabel('残差');
    grid on;
end