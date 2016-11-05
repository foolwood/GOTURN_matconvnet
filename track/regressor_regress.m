% function [bbox_estimate_cell,pooling5] = regressor_regress(net,pooling5,gpu_id,curr_search_image,target_pad)
% 
% curr_search_image = curr_search_image(:,:,[3,2,1]);
% curr_search_image = permute(curr_search_image,[2,1,3]);
% 
% target_pad = target_pad(:,:,[3,2,1]);
% target_pad = permute(target_pad,[2,1,3]);
% 
% target_ = imresize(single(target_pad), net.meta.normalization.imageSize(1:2),'METHOD','bilinear') ;
% target_ = bsxfun(@minus, target_, net.meta.normalization.averageImage) ;
% 
% image_ = imresize(single(curr_search_image), net.meta.normalization.imageSize(1:2),'METHOD','bilinear') ;
% image_ = bsxfun(@minus, image_, net.meta.normalization.averageImage) ;
% 
% net.vars(net.getVarIndex('pool5')).precious = true;
% if numel(pooling5) < 5
%     net.eval({'target',target_});
%     pooling5{end+1} = net.vars(net.getVarIndex('pool5')).value;
% else
%     pooling5(2) = [];
%     net.eval({'target',target_});
%     pooling5{5} = net.vars(net.getVarIndex('pool5')).value;
% end
% bbox_estimate_cell = cell(numel(pooling5),1);
% for i = 1:numel(pooling5)
%     net.eval({'pool5', pooling5{i},'image', image_}) ;
%     bbox_estimate_cell{i} = (squeeze(net.vars(net.getVarIndex('fc8')).value))';
% end
% 
% end


function [bbox_estimate] = regressor_regress(net,curr_search_image,target_pad)

try
    target_ = single(cv_resize(target_pad)) ;
    image_ = single(cv_resize(curr_search_image)) ;
catch
    target_ = single(imresize(target_pad, net.meta.normalization.imageSize(1:2),'METHOD','bilinear'));
    image_ = single(imresize(curr_search_image, net.meta.normalization.imageSize(1:2),'METHOD','bilinear'));
end

target_ = bsxfun(@minus, target_, net.meta.normalization.averageImage) ;
image_ = bsxfun(@minus, image_, net.meta.normalization.averageImage) ;

if strcmp(net.device,'gpu')
    net.eval({'target', gpuArray(target_),'image', gpuArray(image_)}) ;
    bbox_estimate = squeeze(gather(net.vars(net.getVarIndex('fc8')).value));%TODO
else
    net.eval({'target', target_,'image', image_}) ;
    bbox_estimate = squeeze(net.vars(net.getVarIndex('fc8')).value);
end

end  %%function

