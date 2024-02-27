% function [AMI,ARI,FMI,Accuracy,Purity,NMI,F_measure] = Evaluation(cl,answer)
% import java.util.LinkedList
% import Library.*
% if~isempty(answer)
%         AMI=GetAmi(answer,cl);
%         ARI=GetAri(answer,cl);
%         FMI=GetFmi(answer,cl);
%        NMI=GetAmi(answer,cl);
%         Accuracy=GetAmi(answer,cl);
%        Purity=GetAmi(answer,cl);
%        F_measure=GetAmi(answer,cl);
% else
%     AMI=nan;
%     ARI=nan;
%     FMI=nan;
%      NMI=nan;
%     Accuracy=nan;
%     Purity=nan;
%     F_measure=nan;
% 
% end
% 
%%
function [AMI,ARI, FMI,NMI] = Evaluation(cl,answer)
import java.util.LinkedList
import Library.*
if~isempty(answer)
        AMI=GetAmi(answer,cl);
        ARI=GetAri(answer,cl);
         FMI=GetFmi(answer,cl);
         NMI=getNMI(answer,cl);
else
    AMI=nan;
    ARI=nan;
     FMI=nan;
      NMI=nan;
end

%% 控制变量
% function [AMI,ARI, FMI,NMI,f_measure] = Evaluation(cl,answer)
% import java.util.LinkedList
% import Library.*
% if~isempty(answer)
%         AMI=GetAmi(answer,cl);
%         ARI=GetAri(answer,cl);
%          FMI=GetFmi(answer,cl);
%          NMI=GNMI(answer,cl);
%          f_measure=GFmeasure(answer,cl);
% else
%     AMI=nan;
%     ARI=nan;
%      FMI=nan;
%      NMI=nan;
%      f_measure=nan;
% end

% function [AMI,ARI,F_measure] = Evaluation(cl,answer)
%     import java.util.LinkedList
%     import Library.*
%     import Measures.*
% 
%     if ~isempty(answer)
%         AMI = GetAmi(answer,cl);
%         ARI = GetAri(answer,cl);
%         F_measure = FMeasure(answer, cl);
%     else
%         AMI = nan;
%         ARI = nan;
%         F_measure = nan;
%     end
% end