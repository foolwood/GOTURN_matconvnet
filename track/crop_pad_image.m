function [pad_image,pad_image_location,edge_spacing_x,edge_spacing_y] ...
    = crop_pad_image(bbox_tight,image)

image_width = size(image,2);
image_height = size(image,1);
pad_image_location = compute_corppad_image_location(bbox_tight,image);

roi_left = floor(min(pad_image_location(1),image_width));
roi_top = floor(min(pad_image_location(2),image_height));
roi_width = min(image_width,max(1,ceil(pad_image_location(3))));
roi_height = min(image_height,max(1,ceil(pad_image_location(4))));

cropped_image = image(roi_top:roi_top+roi_height-1,roi_left:roi_left+roi_width-1,:);
output_width = max(ceil(max(1,bbox_tight(3)*2)),roi_width);
output_height = max(ceil(max(1,bbox_tight(4)*2)),roi_height);
output_image = zeros(output_height,output_width,3,'uint8');

bbox_center_x = bbox_tight(1)+bbox_tight(3)/2;
bbox_center_y = bbox_tight(2)+bbox_tight(4)/2;

output_width_temp = max(1,bbox_tight(3)*2);
output_height_temp = max(1,bbox_tight(4)*2);
edge_spacing_x = floor(min(max(1,output_width_temp/2-bbox_center_x),size(output_image,2)));
edge_spacing_y = floor(min(max(1,output_height_temp/2-bbox_center_y),size(output_image,1)));


output_image(edge_spacing_y:edge_spacing_y+roi_height-1,...
    edge_spacing_x:edge_spacing_x+roi_width-1,:) = cropped_image;

pad_image = output_image;

end %%function


function pad_image_location = compute_corppad_image_location(bbox_tight,image)

bbox_center_x = bbox_tight(1)+bbox_tight(3)/2;
bbox_center_y = bbox_tight(2)+bbox_tight(4)/2;
image_width = size(image,2);
image_height = size(image,1);

output_width = max(1,bbox_tight(3)*2);             %kContextFactor = 2
output_height = max(1,bbox_tight(4)*2);            %kContextFactor = 2

roi_left = max(1,bbox_center_x-output_width/2);
roi_top = max(1,bbox_center_y-output_height/2);

left_half = min(output_width/2,bbox_center_x);
right_half = min(output_width/2,image_width - bbox_center_x+1);
roi_width = max(1,left_half+right_half);

top_half = min(output_height/2,bbox_center_y);
bottom_half = min(output_height/2,image_height-bbox_center_y+1);
roi_height = max(1,top_half+bottom_half);

pad_image_location = [roi_left,roi_top,roi_width,roi_height];

end %%function

