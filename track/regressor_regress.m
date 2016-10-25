function bbox_estimate = regressor_regress(net,gpu_id,curr_search_image,target_pad)

%%convert from RGB to BGR and permute width and height
curr_search_image = curr_search_image(:,:,[3,2,1]);
curr_search_image = permute(curr_search_image,[2,1,3]);

%%convert from RGB to BGR and permute width and height
target_pad = target_pad(:,:,[3,2,1]);
target_pad = permute(target_pad,[2,1,3]);

target_ = imresize(single(target_pad), net.meta.normalization.imageSize(1:2),'METHOD','bilinear') ;
target_ = bsxfun(@minus, target_, net.meta.normalization.averageImage) ;

image_ = imresize(single(curr_search_image), net.meta.normalization.imageSize(1:2),'METHOD','bilinear') ;
image_ = bsxfun(@minus, image_, net.meta.normalization.averageImage) ;

net.eval({'target', target_,'image', image_}) ;

if numel(gpu_id)>0
    bbox_estimate = squeeze(net.vars(net.getVarIndex('fc8')).value);%TODO
else
    bbox_estimate = squeeze(net.vars(net.getVarIndex('fc8')).value);
end

end