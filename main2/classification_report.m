% 自定义函数8：分类报告
function report = classification_report(y_true, y_pred)
    confMat = confusionmat(y_true, y_pred);
    precision = diag(confMat) ./ sum(confMat, 1);
    recall = diag(confMat) ./ sum(confMat, 2);
    f1_score = 2 * precision .* recall ./ (precision + recall);
    accuracy = sum(diag(confMat)) / sum(confMat(:));
    
    report = sprintf('              precision    recall  f1-score   support\n\n');
    report = [report, sprintf('       0       %.4f      %.4f      %.4f       %d\n', precision(1), recall(1), f1_score(1), sum(y_true == 0))];
    report = [report, sprintf('       1       %.4f      %.4f      %.4f       %d\n\n', precision(2), recall(2), f1_score(2), sum(y_true == 1))];
    report = [report, sprintf('    accuracy                           %.4f       %d\n', accuracy, length(y_true))];
    report = [report, sprintf('   macro avg       %.4f      %.4f      %.4f       %d\n', mean(precision), mean(recall), mean(f1_score), length(y_true))];
    report = [report, sprintf('weighted avg       %.4f      %.4f      %.4f       %d\n', sum(precision .* sum(confMat, 2)) / length(y_true), ...
                                                                 sum(recall .* sum(confMat, 2)) / length(y_true), ...
                                                                 sum(f1_score .* sum(confMat, 2)) / length(y_true), length(y_true))];
end
