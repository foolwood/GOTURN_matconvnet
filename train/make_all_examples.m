function bbox_gt_scaled = make_all_examples(image_prev,image_curr,bbox_prev,bbox_curr,...
    nsample,video_frame_expDir)

if nargin < 5,nsample = 1;end
if nargin < 6,b_save = false;else b_save = true;end

if(size(image_prev, 3)==1)
    image_prev = repmat(image_prev, [1 1 3]);
end
if(size(image_curr, 3)==1)
    image_curr = repmat(image_curr, [1 1 3]);
end

bbox_gt_scaled = zeros([1,1,4,nsample],'single');%buff

target_pad = crop_pad_image(bbox_prev,image_prev);

[curr_search_region,curr_search_location,edge_spacing_x,edge_spacing_y] ...
    = crop_pad_image(bbox_prev,image_curr);

bbox_gt_recentered = recenter(bbox_curr,curr_search_location,edge_spacing_x,edge_spacing_y);
bbox_gt_scaled(1,1,1:4,1) = scale(bbox_gt_recentered,curr_search_region);

if b_save,
    imwrite(target_pad,[sprintf(video_frame_expDir,0,1),'.jpg'],'Quality',100);%,'Mode','lossless'
    imwrite(curr_search_region,[sprintf(video_frame_expDir,1,1),'.jpg'],'Quality',100);%,'Mode','lossless'
end


for n = 2:nsample
    
    bbox_curr_shift = shift(image_curr,bbox_curr);
    
    [rand_search_region,rand_search_location,edge_spacing_x,edge_spacing_y] ...
        = crop_pad_image(bbox_curr_shift,image_curr);
    
    bbox_gt_recentered = recenter(bbox_curr,rand_search_location,edge_spacing_x,edge_spacing_y);
    bbox_gt_scaled(1,1,1:4,n) = scale(bbox_gt_recentered,rand_search_region); 
if b_save,
    imwrite(target_pad,[sprintf(video_frame_expDir,0,n),'.jpg'],'Quality',100);%,'Mode','lossless'
    imwrite(rand_search_region,[sprintf(video_frame_expDir,1,n),'.jpg'],'Quality',100);%,'Mode','lossless'
end
end
save([sprintf(video_frame_expDir,0,0),'.mat'],'bbox_gt_scaled');

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
    [new_width,new_height,-new_width,-new_height]/2;

end %%function

function lp = laplace_rand(lambda)
u = rand(1)-0.5;
lp = sign(u)*log(1-abs(2*u))/lambda;
end %%function