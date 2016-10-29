function [pad_image,pad_image_location,edge_spacing_x,edge_spacing_y] ...
    = crop_pad_image(bbox_tight,image)

pad_image_location = compute_corppad_image_location(bbox_tight,image);

image_cols = size(image,2);
image_rows = size(image,1);
roi_left = floor(min(pad_image_location(1),image_cols));
roi_top = floor(min(pad_image_location(2),image_rows));
roi_width = floor(min(image_cols,max(1,ceil(pad_image_location(3)-pad_image_location(1)))));
roi_height = floor(min(image_rows,max(1,ceil(pad_image_location(4)-pad_image_location(2)))));

cropped_target = image((roi_top+1):(roi_top+roi_height),(roi_left+1):(roi_left+roi_width),:);

output_width = max(ceil(max(1,(bbox_tight(3)-bbox_tight(1))*2)),roi_width);
output_height = max(ceil(max(1,(bbox_tight(4)-bbox_tight(2))*2)),roi_height);
pad_image = zeros(output_height,output_width,3,'uint8');

bbox_center_x = (bbox_tight(1)+bbox_tight(3))/2;
bbox_center_y = (bbox_tight(2)+bbox_tight(4))/2;

output_width_temp = max(1,(bbox_tight(3)-bbox_tight(1))*2);
output_height_temp = max(1,(bbox_tight(4)-bbox_tight(2))*2);

edge_spacing_x = floor(min(max(0,output_width_temp/2-bbox_center_x),size(pad_image,2)-1));
edge_spacing_y = floor(min(max(0,output_height_temp/2-bbox_center_y),size(pad_image,1)-1));


pad_image((edge_spacing_y+1):(edge_spacing_y+roi_height),...
    (edge_spacing_x+1):(edge_spacing_x+roi_width),:) = cropped_target;

end %%function


function pad_image_location = compute_corppad_image_location(target_2xy,image)

bbox_center_x = (target_2xy(1)+target_2xy(3))/2;
bbox_center_y = (target_2xy(2)+target_2xy(4))/2;
image_width = size(image,2);
image_height = size(image,1);

output_width = max(1,(target_2xy(3)-target_2xy(1))*2);             %kContextFactor = 2
output_height = max(1,(target_2xy(4)-target_2xy(2))*2);            %kContextFactor = 2

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

