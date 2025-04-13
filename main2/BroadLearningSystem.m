% 自定义函数7：宽度学习系统分类版
classdef BroadLearningSystem
    properties
        n_feature_nodes
        n_enhance_nodes
        W_e
        beta_e
        W_h
        beta
    end
    
    methods
        function obj = BroadLearningSystem(n_feature_nodes, n_enhance_nodes)
            obj.n_feature_nodes = n_feature_nodes;
            obj.n_enhance_nodes = n_enhance_nodes;
        end
        
        function obj = fit(obj, X, y, learning_rate, epochs, batch_size, loss_type)
            if nargin < 7
                learning_rate = 0.01;
            end
            if nargin < 8
                epochs = 100;
            end
            if nargin < 9
                batch_size = 32;
            end
            if nargin < 10
                loss_type = 'cross_entropy';
            end
            
            n_samples = size(X, 1);
            n_features = size(X, 2);
            
            % 1. 随机初始化特征节点参数
            obj.W_e = 0.1 * randn(n_features, obj.n_feature_nodes);
            obj.beta_e = 0.1 * randn(obj.n_feature_nodes, 1);
            
            % 2. 计算特征节点输出
            Z_e = sigmoid(X * obj.W_e + repmat(obj.beta_e', n_samples, 1));
            
            % 3. 随机初始化增强节点参数
            obj.W_h = 0.1 * randn(obj.n_feature_nodes + n_features, obj.n_enhance_nodes);
            
            % 4. 计算增强节点输出
            Z_h_input = [Z_e, X];
            Z_h = sigmoid(Z_h_input * obj.W_h);
            
            % 5. 构建最终特征矩阵
            Z_final = [Z_h, Z_e, X];
            
            % 6. 初始化输出层权重
            obj.beta = 0.1 * randn(size(Z_final, 2), 1);
            
            % 7. 使用随机梯度下降优化
            for epoch = 1:epochs
                indices = randperm(n_samples);
                
                for i = 1:batch_size:n_samples
                    batch_indices = indices(i:min(i+batch_size-1, n_samples));
                    X_batch = Z_final(batch_indices, :);
                    y_batch = y(batch_indices);
                    
                    % 前向传播
                    logits = X_batch * obj.beta;
                    predictions = sigmoid(logits);
                    
                    % 计算损失和梯度
                    if strcmp(loss_type, 'cross_entropy')
                        error = predictions - y_batch;
                        gradient = X_batch' * error / length(y_batch);
                    elseif strcmp(loss_type, 'cost_sensitive')
                        loss = cost_sensitive_loss(y_batch, predictions);
                        error = (predictions - y_batch) .* (predictions .* (1 - predictions));
                        gradient = X_batch' * error / length(y_batch);
                    elseif strcmp(loss_type, 'focal')
                        gamma = 2;
                        pt = y_batch .* predictions + (1 - y_batch) .* (1 - predictions);
                        alpha = 0.25;
                        dpt = (2 * gamma * (1 - pt).^(gamma - 1) .* pt .* log(pt) + ...
                               (1 - pt).^gamma) .* (y_batch - predictions);
                        gradient = X_batch' * dpt / length(y_batch);
                    end
                    
                    % 更新权重
                    obj.beta = obj.beta - learning_rate * gradient;
                end
                
                % 打印训练进度
                if mod(epoch, 10) == 0
                    preds = obj.predict(X);
                    acc = mean(preds == y);
                    fprintf('Epoch %d, Accuracy: %.4f\n', epoch, acc);
                end
            end
        end
        
        function y_pred = predict(obj, X)
            % 1. 计算特征节点输出
            Z_e = sigmoid(X * obj.W_e + repmat(obj.beta_e', size(X, 1), 1));
            
            % 2. 计算增强节点输出
            Z_h_input = [Z_e, X];
            Z_h = sigmoid(Z_h_input * obj.W_h);
            
            % 3. 构建最终特征矩阵
            Z_final = [Z_h, Z_e, X];
            
            % 4. 计算预测概率
            logits = Z_final * obj.beta;
            probabilities = sigmoid(logits);
            
            % 5. 转换为二分类标签
            y_pred = probabilities >= 0.5;
        end
    end
end


