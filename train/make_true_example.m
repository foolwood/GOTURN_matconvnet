function [image,target,bbox_gt_scaled] =...
    make_true_example(image_prev,image_curr,bbox_prev,bbox_curr,sz)

target_pad = crop_pad_image(bbox_prev,image_prev);

curr_prior_tight = bbox_prev;

[curr_search_region,curr_search_location,edge_spacing_x,edge_spacing_y] ...
    = crop_pad_image(curr_prior_tight,image_curr);

bbox_gt_recentered = recenter(bbox_curr,curr_search_location,edge_spacing_x,edge_spacing_y);
bbox_gt_scaled = scale(bbox_gt_recentered,curr_search_region);

target = imresize(target_pad,sz);  %sz for easy get
image = imresize(target_pad,sz);

% target = cv_resize(target_pad);  %opencv same
% image = cv_resize(target_pad);

end


function bbox_recentered = recenter(bbox_gt,search_location,edge_spacing_x,edge_spacing_y)

bbox_recentered = zeros(1,4);
bbox_recentered(1) = bbox_gt(1) - search_location(1)+edge_spacing_x;
bbox_recentered(2) = bbox_gt(2) - search_location(2)+edge_spacing_y;
bbox_recentered(3) = bbox_gt(3) - search_location(1)+edge_spacing_x;
bbox_recentered(4) = bbox_gt(4) - search_location(2)+edge_spacing_y;

end %%function

function bbox_scaled = scale(bbox_recentered,image)

bbox_scaled = bbox_recentered;
width = size(image,2);
height = size(image,1);
bbox_scaled(1) = bbox_scaled(1)/width;
bbox_scaled(2) = bbox_scaled(2)/height;
bbox_scaled(3) = bbox_scaled(3)/width;
bbox_scaled(4) = bbox_scaled(4)/height;
bbox_scaled = bbox_scaled*10;                       %kScaleFactor = 10

end %%function