%% 初始化种群  
clear
clc
% f = @(x,y)   20 +  x.^2 + y.^2 - 10*cos(2*pi.*x)  - 10*cos(2*pi.*y) ;%[-5.12 ,5.12 ]
 

% x0 = [1:1:100];
% y0 = x0 ;
% [X,Y] = meshgrid(x0,y0);
% Z =f(X,Y)  ;
% figure(1); mesh(X,Y,Z);  
% colormap(parula(5));

N = 100;                         % 初始种群个数  
d = 2;                          % 可行解维数  
ger = 100;                      % 最大迭代次数       
limit = [1,200];               % 设置位置参数限制  
vlimit = [1, 200];               % 设置速度限制  
w = 0.8;                        % 惯性权重  
c1 = 1.49;                       % 自我学习因子  
c2 = 1.49;                       % 群体学习因子   


x =floor(limit(1) + (  limit( 2 ) -  limit( 1)  ) .* rand(N, d));%初始种群的位置  

v = 200*rand(N, d);                  % 初始种群的速度  
xm = x;                          % 每个个体的历史最佳位置  
ym = zeros(1, d);                % 种群的历史最佳位置  
fxm = ones(N, 1)*inf;               % 每个个体的历史最佳适应度   
fym = inf;                       % 种群历史最佳适应度  
% record = zeros(ger,1);
hold on 
% [X,Y] = meshgrid(x(:,1),x(:,2));
% Z = f( X,Y ) ;
% scatter3( x(:,1),x(:,2) ,main( x(:,1),x(:,2) ),'r*' );
% figure(2)  
record=[];

%% 群体更新  
iter = 1;  
% record = zeros(ger, 1);          % 记录器  
while iter <= ger  
     for i=1:N
        fx(i)=main(x(i,1),x(i,2));
     end
     % fx = main( x(:,1),x(:,2) ) ;% 个体当前适应度     
     for i = 1:N        
        if  fx(i)  <fxm(i) 
            fxm(i) = fx(i);     % 更新个体历史最佳适应度  
            xm(i,:) = x(i,:);   % 更新个体历史最佳位置(取值)  
        end   
     end  
    if   min(fxm)<  fym
        [fym, nmin] = min(fxm);   % 更新群体历史最佳适应度  
        ym = xm(nmin, :);      % 更新群体历史最佳位置  
    end  
    v = ceil(v * w + c1 * rand * (xm - x) + c2 * rand * (repmat(ym, N, 1) - x));% 速度更新  
    % 边界速度处理  
    v(v > vlimit(2)) = vlimit(2);  
    v(v < vlimit(1)) = vlimit(1);  
    x = x + v;% 位置更新  
    % 边界位置处理  
    x(x > limit(2)) = limit(2);  
    x(x < limit(1)) = limit(1);  
    record(iter) = fym;%最大值记录  
    % subplot(1,2,1)
    % mesh(X,Y,Z)
    % hold on 
    % scatter3( x(:,1),x(:,2) ,f( x(:,1),x(:,2) ) ,'r*');title(['状态位置变化','-迭代次数：',num2str(iter)])
    plot(record);title('最优适应度进化过程')  
    pause(0.01)  
    iter = iter+1; 

end  

% figure(4);mesh(X,Y,Z); hold on 
% scatter3( x(:,1),x(:,2) ,f( x(:,1),x(:,2) ) ,'r*');title('最终状态位置')  
% disp(['最优值：',num2str(fym)]);  
% disp(['变量取值：',num2str(ym)]);  
