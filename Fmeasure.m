function [AMI_, Accuracy, Purity, FMeasure, Entropy] = CalculateMetrics(true_mem, mem)
    % 计算 AMI
    if nargin==1
        T=true_mem; %contingency table pre-supplied
    elseif nargin==2
        %build the contingency table from membership arrays
        R=max(true_mem);
        C=max(mem);
        n=length(mem);N=n;

        %identify & removing the missing labels
        list_t=ismember(1:R,true_mem);
        list_m=ismember(1:C,mem);
        T=Contingency(true_mem,mem);
        T=T(list_t,list_m);
    end

    n=sum(sum(T));N=n;
    C=T;
    nis=sum(sum(C,2).^2);       %sum of squares of sums of rows
    njs=sum(sum(C,1).^2);       %sum of squares of sums of columns

    t1=nchoosek(n,2);        %total number of pairs of entities
    t2=sum(sum(C.^2));      %sum over rows & columnns of nij^2
    t3=.5*(nis+njs);

    %Expected index (for adjustment)
    nc=(n*(n^2+1)-(n+1)*nis-(n+1)*njs+2*(nis*njs)/n)/(2*(n-1));

    A=t1+t2-t3;         %no. agreements
    D=  -t2+t3;         %no. disagreements

    if t1==nc
       AR=0;            %avoid division by zero; if k=1, define Rand = 0
    else
       AR=(A-nc)/(t1-nc);       %adjusted Rand - Hubert & Arabie 1985
    end

    RI=A/t1;            %Rand 1971        %Probability of agreement
    MIRKIN=D/t1;        %Mirkin 1970    %p(disagreement)
    HI=(A-D)/t1;        %Hubert 1977    %p(agree)-p(disagree)
    Dri=1-RI;           %distance version of the RI
    Dari=1-AR;          %distance version of the ARI

    % update the true dimensions
    [R C]=size(T);
    if C>1
        a=sum(T');
    else
        a=T';
    end
    if R>1
        b=sum(T);
    else
        b=T;
    end

    % 计算熵
    Ha=-(a/n)*log(a/n)'; 
    Hb=-(b/n)*log(b/n)';

    %calculate the MI (unadjusted)
    MI=0;
    for i=1:R
        for j=1:C
            if T(i,j)>0
                MI=MI+T(i,j)*log(T(i,j)*n/(a(i)*b(j)));
            end
        end
    end
    MI=MI/n;

    % 计算准确率
    Correct = sum(max(C, [], 2));
    Accuracy = Correct / N;

    % 计算纯度
    MaxP = max(C, [], 2);
    Purity = sum(MaxP) / N;

    % 计算 F-Measure
    CP=C;
    Pid=ones(length(true_mem),1)*true_mem==ones(length(unique(true_mem)),1)*unique(true_mem)';
    Pj = sum(CP,1);
    Ci = sum(CP,2);
    precision = CP./(Ci*ones(1,C_size));
    recall = CP./(ones(R,1)*Pj);
    F = 2*precision.*recall./(precision+recall);
    FMeasure = sum((Pj./N).*max(F));
    
    % 返回结果
    AMI_ = AR;
    Entropy = Ha + Hb;

end