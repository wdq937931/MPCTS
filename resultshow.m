%     function resultshow(data,CL)
%     
%     PtSize = 2;
%     
%     NC = length(unique(CL));
%     label_set = unique(CL);
%     
%     
%     [N,M] = size(data);
%     
%     figure('Position',[350 350 350 300]);
%     cmap = UiGetColormap(NC);
%     
%     for i=1:NC
%         l=label_set(i);
%         if M~=3
%             if l~=0
%                 scatter(data((CL==l),1),data((CL==l),2),PtSize+5,'o','filled','MarkerFaceColor',cmap(l,:),'MarkerEdgeColor',cmap(l,:));
%             else
%                 scatter(data((CL==l),1),data((CL==l),2),PtSize+50,'x','MarkerEdgeColor','k');
%             end
%         else
%             if l~=0
%                 scatter3(data((CL==l),1),data((CL==l),2),data((CL==l),3),PtSize+5,'o','filled','MarkerFaceColor',cmap(l,:),'MarkerEdgeColor',cmap(l,:));
%             else
%                 scatter3(data((CL==l),1),data((CL==l),2),data((CL==l),3),PtSize+5,'x','filled','MarkerEdgeColor','k');
%             end
%         end
%         hold on
%     end
%     
%     
%     set(gca,'XTickLabel','');
%     set(gca,'YTickLabel','');
%     set(gca,'ZTickLabel','');
%     if M~=3
%         axis off
%     end
%     
%     
%     function [cmap]=UiGetColormap(NC)
%     colormap jet
%     cmap=colormap;
%     cmap=cmap(round(linspace(1,length(cmap),NC+1)),:);
%     cmap=cmap(1:end-1,:);
% 


% function resultshow(data, CL)
%     PtSize = 2;
%     NC = length(unique(CL));
%     label_set = unique(CL);
%     
%     figure('Position',[350 350 350 300]);
%     cmap = jet(NC);
%     
%     for i = 1:NC
%         l = label_set(i);
%         if l ~= 0
%             scatter(data(CL==l, 1), data(CL==l, 2), PtSize+5, cmap(i, :), 'filled');
%         else
%             scatter(data(CL==l, 1), data(CL==l, 2), PtSize+50, 'k', 'x');
%         end
%         hold on;
%         legendCell = cellstr(num2str((1:NC)', 'Subcluster %d'));
%         legendFontSize = 8;  % 设置图例字体大小
%         legendIconSize = 8;   % 设置图例图标大小
%         legend(legendCell, 'Location', 'best');
%     end
%     
%     set(gca, 'XTickLabel', 'auto');
%     set(gca, 'YTickLabel', 'auto');
%    
%     hold off;
% end
function resultshow(data, CL)
    PtSize = 2;
    NC = length(unique(CL));
    label_set = unique(CL);
    
    figure('Position',[350 350 350 300]);
    cmap = jet(NC);
    
    for i = 1:NC
        l = label_set(i);
        if l ~= 0
            scatter(data(CL==l, 1), data(CL==l, 2), PtSize+5, cmap(i, :), 'filled');
        else
            scatter(data(CL==l, 1), data(CL==l, 2), PtSize+50, 'k', 'x');
        end
        hold on;
        legendCell = cellstr(num2str((1:NC)', 'Subcluster %d'));
        legendFontSize = 8;  % 设置图例字体大小
        legendIconSize = 8;   % 设置图例图标大小
        legend(legendCell, 'Location', 'best');
    end
    
    x_min = min(data(:, 1));
    x_max = max(data(:, 1));
    y_min = min(data(:, 2));
    y_max = max(data(:, 2));
    
    n_ticks = 6;
    x_tick_step = (x_max - x_min) / (n_ticks - 1);
    x_ticks = x_min:x_tick_step:x_max;
    y_tick_step = (y_max - y_min) / (n_ticks - 1);
    y_ticks = y_min:y_tick_step:y_max;
    
    x_tick_labels = num2str(round(x_ticks'), '%d');
    y_tick_labels = num2str(round(y_ticks'), '%d');
    set(gca, 'XTick', x_ticks, 'XTickLabel', x_tick_labels);
    set(gca, 'YTick', y_ticks, 'YTickLabel', y_tick_labels);
    
    hold off;
end