
%% 
clear all;
close all;
clc;

%% load dataset
load  dataset/spiral.mat
data =spiral;
answer = data(:,end);
data = data(:,1:end-1);

 
k =6;
C =3;


numRepeats = 1;
Results = cell(numRepeats, 1);
sumRuntime=0;

%% run the code for  times
for i = 1:numRepeats
    fprintf('Iteration %d\n', i);
    
    %% call function
    [cl,runtime] = MPCTS(data, k, C);
    
    %% evaluation
    [AMI,ARI,FMI,NMI] = Evaluation(cl, answer);
    
   
    resultshow(data,cl);
    %% 绘制子簇
sumRuntime = sumRuntime + runtime;
    
    %% save result
    Result = struct;
    Result.k = k;
    Result.C = C;
    Result.AMI = AMI;
  
   Result.ARI = ARI;
    Result.FMI = FMI;
    Result.NMI = NMI;
    Result.runtime = runtime;
    Results{i} = Result;
  
    
end
% diary result %将输出保存为文档
% Print the results
avgRuntime = sumRuntime / numRepeats;
for i = 1:numRepeats
    fprintf('Result for iteration %d:\n', i);
    disp(Results{i});
end
disp(['Average Runtime: ', num2str(avgRuntime) ]);
% diary off
disp(['NMI:', num2str(NMI)])

