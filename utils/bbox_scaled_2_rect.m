function rect = bbox_scaled_2_rect(bbox_scaled,...
    curr_search_region,image_curr,search_location,edge_spacing_x,edge_spacing_y)

%%unscale the estimation to the real image size
bbox_estimate_unscaled = bb_unscale(bbox_scaled,curr_search_region);
%%find the estimated bounding box location relative to the current crop
bbox_estimate_uncentered = bb_uncenter(bbox_estimate_unscaled,image_curr,...
    search_location,edge_spacing_x,edge_spacing_y);

rect = bbox_2_rect(bbox_estimate_uncentered);
end