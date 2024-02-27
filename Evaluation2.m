function [AMI,ARI,FMI] = Evaluation2(cl,answer)
import java.util.LinkedList
import Library.*
if~isempty(answer)
        AMI=GetAmi2(answer,cl);
        ARI=GetAri2(answer,cl);
        FMI=GetFmi2(answer,cl);
        
else
    AMI=nan;
    ARI=nan;
    FMI=nan;
  
end
% %% 引入新的度量指标
% function [AMI, ARI, FMI, ACC, Purity] = Evaluation2(cl, answer)
%     import java.util.LinkedList
%     import Library.*
%     
%     if ~isempty(answer)
%         AMI = GetAmi2(answer, cl);
%         ARI = GetAri2(answer, cl);
%         FMI = GetFmi2(answer, cl);
%         
%         % 计算ACC和Purity
%         [ACC, Purity] = computeMetrics(cl, answer);
%     else
%         AMI = nan;
%         ARI = nan;
%         FMI = nan;
%         ACC = nan;
%         Purity = nan;
%     end
% end
% 
% function [ACC, Purity] = computeMetrics(Mem1, Mem2)
%     % 假设Mem1和Mem2已经存在，即聚类结果和真实标签
%     if nargin < 2 || numel(Mem1) ~= numel(Mem2)
%         error('computeMetrics: 需要长度相等的两个向量作为输入')
%     end
% 
%     N = numel(Mem1); % 样本总数
%     K = max([Mem1, Mem2]); % 聚类簇类别数
% 
%     % 构建混淆矩阵
%     C = zeros(K, K);
%     for i = 1:N
%         C(Mem1(i), Mem2(i)) = C(Mem1(i), Mem2(i)) + 1;
%     end
% 
%     % 计算ACC
%     matchCount = sum(max(C, [], 2)); % 匹配数为混淆矩阵每行的最大值之和
%     ACC = matchCount / N;
% 
%     % 计算Purity
%     clusterSum = sum(C, 2); % 每个簇中样本的总数
%     puritySum = max(C, [], 2); % 混淆矩阵每行的最大值，表示每个簇中最多样本来自的类别数
%     Purity = sum(puritySum) / N;
% end
% %% 
% function Cont=Contingency(Mem1,Mem2)
% 
% if nargin < 2 || min(size(Mem1)) > 1 || min(size(Mem2)) > 1
%    error('Contingency: Requires two vector arguments')
%    return
% end
% 
% Cont=zeros(max(Mem1),max(Mem2));
% 
% for i = 1:length(Mem1);
%    Cont(Mem1(i),Mem2(i))=Cont(Mem1(i),Mem2(i))+1;
% end
% end