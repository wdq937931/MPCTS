function [NMI_, AMI_] = getNMI(true_mem, mem)
    if nargin == 1
        T = true_mem; %预供的列联表
    elseif nargin == 2
        % 从成员数组构建列联表
        R = max(true_mem);
        C = max(mem);
        n = length(mem);
        N = n;

        % 识别并移除缺失的标签
        list_t = ismember(1:R, true_mem);
        list_m = ismember(1:C, mem);
        T = Contingency(true_mem, mem);
        T = T(list_t, list_m);
    end

    % 计算兰德指数和其他指标
    n = sum(sum(T));
    N = n;
    C = T;
    nis = sum(sum(C, 2).^2); % 行和的平方和
    njs = sum(sum(C, 1).^2); % 列和的平方和

    t1 = nchoosek(n, 2); % 实体对的总数
    t2 = sum(sum(C.^2)); % 行与列的平方和
    t3 = 0.5 * (nis + njs);

    % 期望指数（用于调整）
    nc = (n * (n^2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1));

    A = t1 + t2 - t3; % 协议数
    D = -t2 + t3; % 不一致数

    if t1 == nc
        AR = 0; % 避免除以零，如果 k = 1，则定义 Rand = 0
    else
        AR = (A - nc) / (t1 - nc); % 调整后的兰德指数（Adjusted Rand Index）
    end

    RI = A / t1; % 兰德指数（Rand Index）
    MIRKIN = D / t1; % Mirkin 指数（p(disagreement)）
    HI = (A - D) / t1; % Hubert 指数（p(agree) - p(disagree)）
    Dri = 1 - RI; % RI 的距离版本
    Dari = 1 - AR; % ARI 的距离版本

    % 更新真实维度
    [R, C] = size(T);
    if C > 1
        a = sum(T');
    else
        a = T';
    end
    if R > 1
        b = sum(T);
    else
        b = T;
    end

    % 计算熵
    Ha = -(a / n) * log(a / n)';
    Hb = -(b / n) * log(b / n)';

    % 计算互信息
    MI = 0;
    for i = 1:R
        for j = 1:C
            if T(i, j) > 0
                MI = MI + T(i, j) * log(T(i, j) * n / (a(i) * b(j)));
            end
        end
    end
    MI = MI / n;

    % 校正一致性
    AB = a' * b;
    bound = zeros(R, C);
    sumPnij = 0;

    E3 = (AB / n^2) * log(AB / n^2);

    EPLNP = zeros(R, C);
    LogNij = log([1:min(max(a), max(b))] / N);
    for i = 1:R
        for j = 1:C
            sumPnij = 0;
            nij = max(1, a(i) + b(j) - N);
            X = sort([nij, N - a(i) - b(j) + nij]);
            if N - b(j) > X(2)
                nom = [[a(i) - nij + 1:a(i)], [b(j) - nij + 1:b(j)], [X(2) + 1:N - b(j)]];
                dem = [[N - a(i) + 1:N], [1:X(1)]];
            else
                nom = [[a(i) - nij + 1:a(i)], [b(j) - nij + 1:b(j)]];
                dem = [[N - a(i) + 1:N], [N - b(j) + 1:X(2)], [1:X(1)]];
            end
            p0 = prod(nom ./ dem) / N;

            sumPnij = p0;

            EPLNP(i, j) = nij * LogNij(nij) * p0;
            p1 = p0 * (a(i) - nij) * (b(j) - nij) / (nij + 1) / (N - a(i) - b(j) + nij + 1);

            for nij = max(1, a(i) + b(j) - N) + 1:1:min(a(i), b(j))
                sumPnij = sumPnij + p1;
                EPLNP(i, j) = EPLNP(i, j) + nij * LogNij(nij) * p1;
                p1 = p1 * (a(i) - nij) * (b(j) - nij) / (nij + 1) / (N - a(i) - b(j) + nij + 1);

            end
            CC = N * (a(i) - 1) * (b(j) - 1) / a(i) / b(j) / (N - 1) + N / a(i) / b(j);
            bound(i, j) = a(i) * b(j) / N^2 * log(CC);
        end
    end

    EMI_bound = sum(sum(bound));
    EMI_bound_2 = log(R * C / N + (N - R) * (N - C) / (N * (N - 1)));
    EMI = sum(sum(EPLNP - E3));

    AMI_ = (MI - EMI) / (max(Ha, Hb) - EMI);
    NMI_ = MI / sqrt(Ha * Hb);

    % 如果期望互信息微不足道，则使用归一化互信息
    if abs(EMI) > EMI_bound
        AMI_ = NMI_;
    end
end

function Cont = Contingency(Mem1, Mem2)
    if nargin < 2 || min(size(Mem1)) > 1 || min(size(Mem2)) > 1
        error('Contingency: 需要两个向量参数');
        return
    end

    Cont = zeros(max(Mem1), max(Mem2));

    for i = 1:length(Mem1)
        Cont(Mem1(i), Mem2(i)) = Cont(Mem1(i), Mem2(i)) + 1;
    end
end