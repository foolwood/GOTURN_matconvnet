function bbox_estimate_unscaled = bb_unscale(bbox_estimate,curr_search_region)

image_height = size(curr_search_region,1);
image_width = size(curr_search_region,2);
bbox_estimate_unscaled = bbox_estimate;
bbox_estimate_unscaled = bbox_estimate_unscaled/10.;     %kScaleFactor = 10;
bbox_estimate_unscaled([1,3]) = bbox_estimate_unscaled([1,3])*image_width;
bbox_estimate_unscaled([2,4]) = bbox_estimate_unscaled([2,4])*image_height;

end %%function