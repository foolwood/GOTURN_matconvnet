function [result_rect, time] = tracker(img_files, ground_truth, net, gpu_id, show_visualization)
%% speed up 
%   image_bank = vl_imreadjpeg(img_files,'NumThreads', 4);
%   image_prev = image_bank{1};
%   image_curr = image_bank{frame};
%
%%

close all
if show_visualization,  %create video interface
    update_visualization = show_video(img_files);
end

x_minmax = minmax(ground_truth(1,1:2:end));
y_minmax = minmax(ground_truth(1,2:2:end));
bbox_gt = [x_minmax(1),y_minmax(1),x_minmax(2),y_minmax(2)]-1;%zero-index

time = 0;
result_rect = bsxfun(@times,bbox_2_rect(bbox_gt),...
    ones(numel(img_files),1));  %to calculate precision

image_prev = imread(img_files{1});
bbox_prev_tight = bbox_gt;
bbox_prev_prior_tight = bbox_gt;

for frame = 2:numel(img_files),
  
    image_curr = imread(img_files{frame});
    tic;
    target_pad = crop_pad_image(bbox_prev_tight,image_prev);
    [curr_search_region,search_location,edge_spacing_x,...
        edge_spacing_y] = crop_pad_image(bbox_prev_prior_tight,image_curr);
    
    bbox_estimate = regressor_regress(net,gpu_id,curr_search_region,target_pad);
    
    %%unscale the estimation to the real image size
    bbox_estimate_unscaled = bb_unscale(bbox_estimate,curr_search_region);
    %%find the estimated bounding box location relative to the current crop
    bbox_estimate_uncentered = bb_uncenter(bbox_estimate_unscaled,image_curr,...
        search_location,edge_spacing_x,edge_spacing_y);
    
    image_prev = image_curr;
    bbox_prev_tight = bbox_estimate_uncentered;
    bbox_prev_prior_tight = bbox_estimate_uncentered;%TODO
    
    result_rect(frame,:) = bbox_2_rect(bbox_estimate_uncentered);
    
    time = time + toc;
    
    if show_visualization,
        bbox_estimate_uncentered_cell = [];
        stop = update_visualization(frame, ground_truth(frame,:),result_rect(frame,:),bbox_estimate_uncentered_cell);
        if stop, break, end
    end
    
end

end %%function
