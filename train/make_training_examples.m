function [image,target,bbox_gt_scaled] =...
    make_training_examples(image_prev,image_curr,bbox_prev,bbox_curr,sz)

target_pad = crop_pad_image(bbox_prev,image_prev);


bbox_curr_shift = shift(image_curr,bbox_curr);

[rand_search_region,rand_search_location,edge_spacing_x,edge_spacing_y] ...
    = crop_pad_image(bbox_curr_shift,image_curr);

bbox_gt_recentered = recenter(bbox_curr,rand_search_location,edge_spacing_x,edge_spacing_y);
bbox_gt_scaled = scale(bbox_gt_recentered,rand_search_region);

target = imresize(target_pad,sz);  %sz for easy get
image = imresize(rand_search_region,sz);

% target = cv_resize(target_pad);  %opencv same
% image = cv_resize(rand_search_region);

end


function bbox_recentered = recenter(bbox_gt,search_location,edge_spacing_x,edge_spacing_y)

bbox_recentered = zeros(1,4,'single');
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



function bbox_curr_shift = shift(image_curr,bbox_curr)
width = bbox_curr(3) - bbox_curr(1);
height = bbox_curr(4) - bbox_curr(2);
center_x = (bbox_curr(1) + bbox_curr(3))/2;
center_y = (bbox_curr(2) + bbox_curr(4))/2;

width_scale_factor = max(min(laplace_rand(15),0.4),-0.4);
new_width = min(max(width*(1+width_scale_factor),1),size(image_curr,2)-1);

height_scale_factor = max(min(laplace_rand(15),0.4),-0.4);
new_height = min(max(height*(1+height_scale_factor),1),size(image_curr,1)-1);

new_x_temp = center_x+laplace_rand(5);
new_center_x = min(size(image_curr,2)-new_width/2,max(new_width/2,new_x_temp));
new_center_x = min(max(new_center_x,center_x-width),center_x+width);

new_y_temp = center_y+laplace_rand(5);
new_center_y = min(size(image_curr,1)-new_height/2,max(new_height/2,new_y_temp));
new_center_y = min(max(new_center_y,center_y-height),center_y+height);

bbox_curr_shift = [new_center_x,new_center_y,new_center_x,new_center_y]-...
    [new_width,new_height,new_width,new_height]/2;

end %%function

function lp = laplace_rand(lambda)
u = rand(1)-0.5;
lp = log(1-abs(2*u))/lambda;
end %%function