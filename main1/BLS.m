% BLS类
classdef BLS
    properties
        n_feature_nodes
        n_enhance_nodes
        W_e
        beta_e
        W_h
        beta
    end
    
    methods
        % 构造函数
        function obj = BLS(n_feature_nodes, n_enhance_nodes)
            obj.n_feature_nodes = n_feature_nodes;
            obj.n_enhance_nodes = n_enhance_nodes;
        end
        
        % Sigmoid激活函数
        function output = sigmoid(obj, x)
            output = 1 ./ (1 + exp(-x));
        end
        
        % 训练函数
        function obj = fit(obj, X, y)
            [n_samples, n_features] = size(X);
            
            % 1. 随机初始化特征节点参数
            obj.W_e = randn(n_features, obj.n_feature_nodes);
            obj.beta_e = randn(1, obj.n_feature_nodes);
            
            % 2. 计算特征节点输出
            Z_e = obj.sigmoid(X * obj.W_e + repmat(obj.beta_e, n_samples, 1));
            
            % 3. 随机初始化增强节点参数
            obj.W_h = randn(n_features + obj.n_feature_nodes, obj.n_enhance_nodes);
            
            % 4. 计算增强节点输出
            Z_e_X = [Z_e, X];
            Z_h = obj.sigmoid(Z_e_X * obj.W_h);
            
            % 5. 计算输出层权重(伪逆法)
            H = [Z_h, Z_e, X];
            obj.beta = pinv(H) * y;
        end
        
        % 预测函数
        function y_pred = predict(obj, X)
            % 1. 计算特征节点输出
            Z_e = obj.sigmoid(X * obj.W_e + repmat(obj.beta_e, size(X, 1), 1));
            
            % 2. 计算增强节点输出
            Z_e_X = [Z_e, X];
            Z_h = obj.sigmoid(Z_e_X * obj.W_h);
            
            % 3. 计算预测输出
            H = [Z_h, Z_e, X];
            y_pred = H * obj.beta;
        end
    end
end
