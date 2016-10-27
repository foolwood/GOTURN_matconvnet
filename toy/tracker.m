function tracker(img_files, ground_truth)
%% speed up 
%   image_bank = vl_imreadjpeg(img_files,'NumThreads', 4);
%   image_prev = image_bank{1};
%   image_curr = image_bank{frame};
%
%%


close all
update_visualization = show_video(img_files);

x_minmax = minmax(ground_truth(:,1:2:end));
y_minmax = minmax(ground_truth(:,2:2:end));
bbox_gt = [x_minmax(:,1),y_minmax(:,1),x_minmax(:,2)-x_minmax(:,1),y_minmax(:,2)-y_minmax(:,1)];


for frame = 2:numel(img_files),
  
    bbox_estimate_uncentered_cell = [];
    stop = update_visualization(frame, ground_truth(frame,:),bbox_gt(frame,:),bbox_estimate_uncentered_cell);
    if stop, break, end
    drawnow
end

end %%function

