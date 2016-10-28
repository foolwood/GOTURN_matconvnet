function [search_2xy_in_net,pad_target,pad_search] = crop_pair_pad_image(target,search,target_2xy,search_2xy)

pad_image_location = compute_corppad_image_location(target_2xy,target);

image_cols = size(target,2);
image_rows = size(target,1);
roi_left = floor(min(pad_image_location(1),image_cols))+1;
roi_top = floor(min(pad_image_location(2),image_rows))+1;
roi_width = floor(min(image_cols,max(1,ceil(pad_image_location(3)-pad_image_location(1)))));
roi_height = floor(min(image_rows,max(1,ceil(pad_image_location(4)-pad_image_location(2)))));

cropped_target = target(roi_top:roi_top+roi_height-1,roi_left:roi_left+roi_width-1,:);
cropped_search = search(roi_top:roi_top+roi_height-1,roi_left:roi_left+roi_width-1,:);

output_width = max(ceil(max(1,(target_2xy(3)-target_2xy(1))*2)),roi_width);
output_height = max(ceil(max(1,(target_2xy(4)-target_2xy(2))*2)),roi_height);
pad_target = zeros(output_height,output_width,3,'uint8');
pad_search = zeros(output_height,output_width,3,'uint8');

bbox_center_x = (tight_2xy(1)+tight_2xy(3))/2;
bbox_center_y = (tight_2xy(2)+tight_2xy(4))/2;

output_width_temp = max(1,(target_2xy(3)-target_2xy(1))*2);
output_height_temp = max(1,(target_2xy(4)-target_2xy(2))*2);

edge_spacing_x = floor(min(max(0,output_width_temp/2-bbox_center_x),size(pad_target,2)-1));
edge_spacing_y = floor(min(max(0,output_height_temp/2-bbox_center_y),size(pad_target,1)-1));


pad_target((edge_spacing_y+1):(edge_spacing_y+roi_height),...
    (edge_spacing_x+1):(edge_spacing_x+roi_width),:) = cropped_target;
pad_search((edge_spacing_y+1):(edge_spacing_y+roi_height),...
    (edge_spacing_x+1):(edge_spacing_x+roi_width),:) = cropped_search;

search_2xy_recentered = recenter(search_2xy,pad_image_location,edge_spacing_x,edge_spacing_y);
search_2xy_in_net = scale(search_2xy_recentered,pad_search);
end %%function


function pad_image_location = compute_corppad_image_location(target_2xy,image)

bbox_center_x = (target_2xy(1)+target_2xy(3))/2;
bbox_center_y = (bbox_tight(2)+bbox_tight(4))/2;
image_width = size(image,2);
image_height = size(image,1);

output_width = max(1,(bbox_tight(3)-target_2xy(1))*2);             %kContextFactor = 2
output_height = max(1,(bbox_tight(4)-target_2xy(2))*2);            %kContextFactor = 2

roi_left = max(0,bbox_center_x-output_width/2);
roi_top = max(0,bbox_center_y-output_height/2);

left_half = min(output_width/2,bbox_center_x);
right_half = min(output_width/2,image_width - bbox_center_x);
roi_width = max(1,left_half+right_half);

top_half = min(output_height/2,bbox_center_y);
bottom_half = min(output_height/2,image_height-bbox_center_y);
roi_height = max(1,top_half+bottom_half);

pad_image_location = [roi_left,roi_top,roi_left+roi_width,roi_top+roi_height];

end %%function


function search_2xy_recentered = recenter(search_2xy,pad_image_location,edge_spacing_x,edge_spacing_y)
search_2xy_recentered = zeros(1,4);
search_2xy_recentered(1) = search_2xy(1) - pad_image_location(1)+edge_spacing_x;
search_2xy_recentered(2) = search_2xy(2) - pad_image_location(2)+edge_spacing_y;
search_2xy_recentered(3) = search_2xy(3) - pad_image_location(1)+edge_spacing_x;
search_2xy_recentered(4) = search_2xy(4) - pad_image_location(2)+edge_spacing_y;

end %%function

function search_2xy_in_net = scale(search_2xy_recentered,pad_search)
search_2xy_in_net = search_2xy_recentered;
width = size(pad_search,2);
height = size(pad_search,1);
search_2xy_in_net(1) = search_2xy_in_net(1)/width;
search_2xy_in_net(2) = search_2xy_in_net(2)/height;
search_2xy_in_net(3) = search_2xy_in_net(3)/width;
search_2xy_in_net(4) = search_2xy_in_net(4)/height;
search_2xy_in_net = search_2xy_in_net*10;                       %kScaleFactor = 10

end

