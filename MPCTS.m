
function [CL, runtime ] = MPCTS(data,k,C)
close all
[n,d]  = size(data); 
%% Normalization
data=(data-min(data))./(max(data)-min(data));
% 计算每列的平均值
mean_values = nanmean(data);
% mean_values = mean(data, 'omitnan');

% 找出缺失值的位置并替换为平均值
missing_values_index = isnan(data);
data(missing_values_index) = mean_values(missing_values_index);

% nan_indices = isnan(data);
% data(nan_indices) = mean_values(nan_indices);
tic;
eta=0.1;
%%  KNN
if d<=9
    [knn,knn_dist] = knnsearch(data,data,'k',k);
else
    %dist = mahal(data, data);
    dist = pdist2(data,data,'euclidean');
    %dist = pdist2(data,data,'cityblock');
    [knn_dist,knn] = sort(dist,2);%2表示按行排序
end
% if d <= 1
%     snn = snnsearch(data, data, k); % 使用SNN搜索算法
%     knn = snn(:, 1:k); % 获取k个最近邻索引
%     knn_dist = zeros(size(knn)); % 创建一个用于存储距离的空数组
%     
%     % 计算每个数据点与其最近邻之间的欧几里德距离
%     for i = 1:size(knn, 1)
%         knn_dist(i, :) = sqrt(sum((data(knn(i, :), :) - data(i, :)).^2, 2));
%     end
% else
%     dist = pdist2(data, data, 'euclidean'); % 计算数据之间的欧几里德距离
%     [knn_dist, knn] = sort(dist, 2); % 按行对距离进行排序，得到最近邻索引
%     
%     % 如果想使用SNN算法替代KNN算法，可以在此处针对knn进行处理
%     % ...
% end
  %% 加权距离计算
% % 计算特征的均值
% % meanVals = mean(data);
% 
% % % % 计算特征的标准差
%   stdVals = std(data);
% %     meanVals = mean(data);
%  weights = stdVals / sum(stdVals);
% % weights = madVals / sum(madVals);
% 
% % 计算新的距离度量矩阵
% new_dist = zeros(size(data, 1));
% for i = 1:size(data, 1)
%     for j = 1:size(data, 1)
%         % 计算新的距离度量
%         diff = data(i, :) - data(j, :);
%         weighted_diff = weights .* diff;
%         distance = sqrt(sum(weighted_diff .^ 2));
% 
%         
%         % 将距离存储在新的距离度量矩阵中
%         new_dist(i, j) = distance;
%     end

%%  局部密度计算 Local density calculation
result = 1 / sqrt(2*pi);
rho1=sum(exp(-knn_dist(:,2:k)).^2,2)*result;

%rho = k*sum(knn_dist(:,2:k).^1,2).^-1;
% 调用calDensity函数计算密度值
%rho = calDensity;
%rho = k*sum(knn_dist(:,2:k).^1,2).^-1;
% 计算每个数据点到其k个最近邻之间的距离的倒数
% dist_inverse = 1 ./ knn_dist(:, 2:k);
% % 计算密度
% rho = k * sum(dist_inverse, 2);
%% 全局密度计算  Global density calculation
global_density = zeros(n, 1); % 存储全局密度

% 对于每个数据点 i，计算其与其他数据点的局部密度之和
for i = 1:n
    global_density(i) = sum(rho1) - rho1(i);
end
%% 融合之后的样本点密度计算公式 
rho=rho1+global_density(i);

%% Find parent node

[~,OrdRho] = sort(rho, 'descend');
omega = zeros(1, n); % omega: depth value

for i = 1:n
    for j = 2:k
        neigh = knn(OrdRho(i), j);
        if(rho(OrdRho(i)) < rho(neigh))
            NPN(OrdRho(i)) = neigh;
            omega(OrdRho(i)) = omega(neigh) + 1;
            break
        end
    end
end
%% find sub-cluster centers
sub_centers = find(omega==0);
n_sc = length(sub_centers);
% generate sub-clsuters
for i=1:n
    sub_L(i)=-1; 
end
sub_L(sub_centers) = (1:n_sc);
for i=1:n
    if (sub_L(OrdRho(i))==-1)
        sub_L(OrdRho(i))=sub_L(NPN(OrdRho(i)));
    end
end

%%  衰减策略的引入
theta = ones(n,1); 
descendant = zeros(n,1);  
[~,OrdRho]=sort(rho,'descend');
for i=1:n
    for j=2:k
        neigh=knn(OrdRho(i),j);
        if(rho(OrdRho(i))<rho(neigh))
            NPN(OrdRho(i))=neigh;%% NPN:neigbor-based parent node
            theta(OrdRho(i)) = theta(neigh)* rho(OrdRho(i))/rho(neigh);
            descendant(neigh) = descendant(neigh)+1;
            break
        end
    end
end
%% 
sl=-1*ones(n,1);
sl(sub_centers) = (1:n_sc); %% give unique sub-labels to density peaks.
for i=1:n
    if (sl(OrdRho(i))==-1)
        sl(OrdRho(i))=sl(NPN(OrdRho(i)));%% inherit sub-labels from NPN
    end
end
for i = 1:n_sc
    child_sub= descendant(sl==i);
    edge(i) = length(find(child_sub==0)); %% edge(i): the edge number of sub-cluster 
end
borderpair = obtain_borderpairs(sl,k,knn,knn_dist);
blink = obtain_borderlinks(borderpair);
% [~, OrdRho] = sort(rho, 'descend');
for i = 1:n
    omega(i) = 0; 
end
for i = 1:n
    for j = 2:k
        neigh = knn(OrdRho(i), j);
        if(rho(OrdRho(i)) < rho(neigh))
            NPN(OrdRho(i)) = neigh;
            omega(OrdRho(i)) = omega(neigh) + 1;
            break
        end
    end
end
sub_centers = find(omega==0);
n_sc = length(sub_centers);
sub_L(sub_centers) = (1:n_sc);
% generate sub-clsuters
for i=1:n
    sub_L(i)=-1; 
end
sub_L(sub_centers) = (1:n_sc);
for i=1:n
    if (sub_L(OrdRho(i))==-1)
        sub_L(OrdRho(i))=sub_L(NPN(OrdRho(i)));
    end
end
%% 衰减策略
theta = ones(n,1); 
descendant = zeros(n,1);
[~, OrdRho] = sort(rho, 'descend');
for i = 1:n
    for j = 2:k
        neigh = knn(OrdRho(i), j);
        if(rho(OrdRho(i)) < rho(neigh))
            NPN(OrdRho(i)) = neigh; 
            theta(OrdRho(i)) = theta(neigh) * rho(OrdRho(i)) / rho(neigh);
            descendant(neigh) = descendant(neigh) + 1;
            break
        end
    end
end

%%  decay strategy
lambda_initial = 0.99; % initial decay factor (scaling factor)
alpha = 0.5; % decay rate factor
PHI = zeros(n, 1); 
for i = 1:n
    
        rho_val = rho(i); % 密度值
        theta_val = theta(i); % 
        descendant_val = descendant(i); % 后代节点数
        depth = omega(i) + 1; % depth value of the current sample point
        lambda = lambda_initial * exp(-alpha * depth); % dynamic decay factor
        PHI(i) =  lambda * theta_val * (1 / (1 + descendant_val)); % 
   
end

if isempty(blink)
    CL = sl';
    NC = n_pk;
    runtime = toc;
    %% show result
    if isshowresult
        resultshow(data,CL);
    end
    return
end

n_blink = size(blink,1);
simimesgs = cell(n_sc,n_sc); 
for i = 1:n_blink
    ii = blink(i,1);
    jj = blink(i,2);
    pk1 = sl(ii);
    pk2 = sl(jj);
    smesgs = simimesgs(pk1,pk2);
    smesgs{1} = [smesgs{1};(PHI(ii)+PHI(jj))/2];
    simimesgs(pk1,pk2) = smesgs;
    simimesgs(pk2,pk1) = smesgs;
end

sim = zeros(n_sc,n_sc);
sim_list = [];
for pk1=1:n_sc-1
    for pk2 =pk1+1:n_sc
        smesgs = simimesgs(pk1,pk2);
        smesgs = smesgs{:};
        max_smesg = max(smesgs);
        min_n_smesg = ceil(min(edge(pk1),edge(pk2))*eta); 
        smesgs = sort([smesgs;zeros(min_n_smesg,1)],'descend');
        smesgs = smesgs(1:min_n_smesg);
        if max_smesg>0
            %Gamma = mean(abs(smesgs - max_smesg))/max_smesg; %%
             Gamma = mean(smesgs); 
                  sim(pk2,pk1) =sum((max_smesg+ Gamma)/2);
                  sim(pk1,pk2) =sum((max_smesg+ Gamma)/2);
        end
        sim_list = [sim_list sim(pk1,pk2)];
    end
end

%% 绘制子簇
% 创建颜色映射

%% Fused sub cluster
avgLink = linkage(1-sim_list,'average');
 F_sub_L = cluster( avgLink ,C); 

%% assign final cluster label
for i=1:n_sc
    AA = find(sub_L==i);
    CL(AA) = F_sub_L(i); %% CL
end
 runtime = toc;
%% draw result
 cmap = colormap;
subplot(2,2,1:2)
dendrogram( avgLink,0);
axis([0 n_sc+1 0 1]);
xlabel ('sub-cluster','FontSize',20.0);
ylabel ('similarity','FontSize',20.0);

title('dendrogram','FontSize',20);
set(gca,'YTickLabel','');
hold on
subplot(2,2,3)
for i=1:n_sc
    ic=int8((i*64.)/(n_sc*1.));
    AA = find(sub_L== i);
    plot(data(AA,1),data(AA,2),'o','MarkerSize',2,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
    text(data(sub_centers(i),1)-0.010,data(sub_centers(i),2), num2str(i),'FontSize',10,'Color','r','FontWeight','Bold');
    hold on
end
set(gca,'XTickLabel','');
set(gca,'YTickLabel','');
% 绘制子簇数据点
% 使用 jet 颜色映射
cmap = jet(n_sc);

% 绘制子簇数据点
figure;
hold on;
for i = 1:n_sc
    AA = find(sub_L == i);
    plot(data(AA, 1), data(AA, 2), 'o', 'MarkerSize', 4, 'MarkerFaceColor', cmap(i, :), 'MarkerEdgeColor', cmap(i, :));
end

% 设置图例
legendCell = cellstr(num2str((1:n_sc)', 'Subcluster %d'));
legend(legendCell, 'Location', 'best');





