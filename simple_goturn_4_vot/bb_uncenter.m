function bbox_estimate_uncentered = bb_uncenter(bbox_estimate_unscaled,...
    image_curr,search_location,edge_spacing_x,edge_spacing_y)

bbox_estimate_uncentered(1) = max(1,bbox_estimate_unscaled(1)+search_location(1)-edge_spacing_x);
bbox_estimate_uncentered(2) = max(1,bbox_estimate_unscaled(2)+search_location(2)-edge_spacing_y);

image_curr_height = size(image_curr,1);
image_curr_width = size(image_curr,2);

bbox_estimate_uncentered(3) = min(image_curr_width,...
    bbox_estimate_unscaled(3)+search_location(1)-edge_spacing_x);
bbox_estimate_uncentered(4) = min(image_curr_height,...
    bbox_estimate_unscaled(4)+search_location(2)-edge_spacing_y);


% change x1y1x2y2->xywh and 1-indexed
bbox_estimate_uncentered(3) = bbox_estimate_uncentered(3)-bbox_estimate_uncentered(1);
bbox_estimate_uncentered(4) = bbox_estimate_uncentered(4)-bbox_estimate_uncentered(2);
bbox_estimate_uncentered(1) = bbox_estimate_uncentered(1)+1;
bbox_estimate_uncentered(2) = bbox_estimate_uncentered(2)+1;
end