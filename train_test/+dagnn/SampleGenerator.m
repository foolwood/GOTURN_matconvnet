%DAGNN.SampleGenerator Generate sample for GOTURN
% input:
%     -bbox_prev_gt       1x4
%     -bbox_curr_gt       1x4
%     -image_prev      HxWx3x1
%     -image_curr      HxWx3x1
% output:
%     -bbox_gt_scaled           1x 1x4xNo
%     -image_target_pad       HoxWox3xNo
%     -image_search_pad       HoxWox3xNo
%   2016 Qiang Wang
classdef SampleGenerator < dagnn.Layer
    
    properties
        Ho = 0;
        Wo = 0;
        No = 1;
        visual = true;
    end
    
    properties (Transient)
        % the grid --> this is cached
        % has the size: [2 x HoWo]
        xxyy ;
        averageImage = reshape(single([123,117,104]),[1,1,3]);
    end
    
    methods
        
        function outputs = forward(obj, inputs, ~)
            bbox_prev_gt = inputs{1};
            bbox_curr_gt = inputs{2};
            image_prev = inputs{3};
            image_curr = inputs{4};
            
            % generate the grid coordinates:
            useGPU = isa(bbox_prev_gt, 'gpuArray');
            if isempty(obj.xxyy)
                obj.initGrid(useGPU);
            end
            
            [im_h,im_w,im_c,~] = size(image_prev);
            if im_c == 1 
                image_prev = repmat(image_prev,[1,1,3,1]);
            end
            
            
            %% target
            target_crop_w = 2*(bbox_prev_gt(3)-bbox_prev_gt(1));
            target_crop_h = 2*(bbox_prev_gt(4)-bbox_prev_gt(2));
            target_crop_cx = (bbox_prev_gt(3)+bbox_prev_gt(1))/2;
            target_crop_cy = (bbox_prev_gt(4)+bbox_prev_gt(2))/2;
            
            cy_t = (target_crop_cy*2/(im_h-1))-1;
            cx_t = (target_crop_cx*2/(im_w-1))-1;
            
            h_s = target_crop_h/(im_h-1);
            w_s = target_crop_w/(im_w-1);
            
            s = reshape([h_s;w_s], 2,1,1); % x,y scaling
            t = reshape([cy_t;cx_t], 2,1,1); % translation
           
            g = bsxfun(@times, obj.xxyy, s); % scale
            g = bsxfun(@plus, g, t); % translate
            g = reshape(g, 2,obj.Ho,obj.Wo,1);
            
            target_pad = vl_nnbilinearsampler(image_prev, single(g));
            
            image_target_pad = repmat(target_pad,[1,1,1,obj.No]);
            image_search_pad = vl_nnbilinearsampler(image_curr, single(g));
            
            if useGPU,
                bbox_gt_scaled = gpuArray(zeros([1,1,4,obj.No],'single'));%buff
            else
                bbox_gt_scaled = zeros([1,1,4,obj.No],'single');%buff
            end
            
            curr_search_location = [target_crop_cx-target_crop_w/2;target_crop_cy-target_crop_h/2];
            bbox_gt_recentered = recenter(bbox_curr_gt',curr_search_location);
            bbox_gt_scaled(1,1,1:4,1) = scale(bbox_gt_recentered,target_crop_w,target_crop_h);
            
            %% search
            if obj.No > 1
                bbox_curr_shift = shift([im_h,im_w],bbox_prev_gt,obj.No-1,useGPU);
                
                target_crop_w = 2*(bbox_curr_shift(3,:)-bbox_curr_shift(1,:));
                target_crop_h = 2*(bbox_curr_shift(4,:)-bbox_curr_shift(2,:));
                target_crop_cx = (bbox_curr_shift(1,:)+bbox_curr_shift(3,:))/2;
                target_crop_cy = (bbox_curr_shift(2,:)+bbox_curr_shift(4,:))/2;
                
                rand_search_location = [target_crop_cx-target_crop_w/2;target_crop_cy-target_crop_h/2];
                bbox_gt_recentered = recenter(bbox_curr_gt',rand_search_location);
                bbox_gt_scaled(1,1,1:4,2:obj.No) = scale(bbox_gt_recentered,target_crop_w,target_crop_h);
                
                
                cy_t = (target_crop_cy*2/(im_h-1))-1;
                cx_t = (target_crop_cx*2/(im_w-1))-1;
                
                h_s = target_crop_h/(im_h-1);
                w_s = target_crop_w/(im_w-1);
                
                s = reshape([h_s;w_s], 2,1,[]); % x,y scaling
                t = reshape([cy_t;cx_t], 2,1,[]); % translation
                
                g = bsxfun(@times, obj.xxyy, s); % scale
                g = bsxfun(@plus, g, t); % translate
                g = reshape(g, 2,obj.Ho,obj.Wo,[]);
                
                image_search_pad(:,:,:,2:obj.No) = vl_nnbilinearsampler(image_curr, single(g));
            end
            
            if obj.visual
                scaled2rect = @(x) [x(1),x(2),x(3)-x(1),x(4)-x(2)]/10*227+1;
                
                for i = 1:obj.No
                    subplot(4,obj.No/4,i);imshow(uint8(image_search_pad(:,:,:,i)));
                    rectangle('Position',scaled2rect(gather(bbox_gt_scaled(1,1,1:4,i))),'EdgeColor',[0 1 0]);
                end
                drawnow;
            end
            
            image_target_pad = bsxfun(@minus,image_target_pad,obj.averageImage);
            image_search_pad = bsxfun(@minus,image_search_pad,obj.averageImage);
            outputs = {bbox_gt_scaled,image_target_pad,image_search_pad};
        end
                
        function obj = SampleGenerator(varargin)
            obj.load(varargin);
            % get the output sizes:
            obj.Ho = obj.Ho;
            obj.Wo = obj.Wo;
            obj.No = obj.No;
            obj.xxyy = [];
            obj.averageImage = obj.averageImage;
            obj.visual = obj.visual;
        end
        
        function obj = reset(obj)
            reset@dagnn.Layer(obj) ;
            obj.xxyy = [] ;
        end
        
        function initGrid(obj, useGPU)
            % initialize the grid:
            % this is a constant
            xi = linspace(-1, 1, obj.Ho);
            yi = linspace(-1, 1, obj.Wo);
            [yy,xx] = meshgrid(yi,xi);
            xxyy_ = [xx(:), yy(:)]' ; % 2xM
            if useGPU
                obj.xxyy = gpuArray(xxyy_);
                obj.averageImage = gpuArray(obj.averageImage);
            end
            obj.xxyy = xxyy_ ;
        end
    end
end

function bbox_recentered = recenter(bbox_gt,search_location)
bbox_recentered= bsxfun(@minus,bbox_gt,search_location([1,2,1,2],:));
end %%function

function bbox_scaled = scale(bbox_recentered,Wo,Ho)
bbox_scaled = bsxfun(@rdivide,bbox_recentered*10,[Wo;Ho;Wo;Ho]);    %kScaleFactor = 10
end %%function

function bbox_curr_shift = shift(image_sz,bbox_curr,n,useGPU)
if useGPU
    bbox_curr_shift = gpuArray(zeros(4,n,'single'));
else
    bbox_curr_shift = zeros(4,n,'single');
end

width = bbox_curr(3) - bbox_curr(1);
height = bbox_curr(4) - bbox_curr(2);
center_x = (bbox_curr(1) + bbox_curr(3))/2;
center_y = (bbox_curr(2) + bbox_curr(4))/2;

for i = 1:n
    width_scale_factor = max(min(laplace_rand(15),0.4),-0.4);
    new_width = min(max(width*(1+width_scale_factor),1),image_sz(2)-1);
    
    height_scale_factor = max(min(laplace_rand(15),0.4),-0.4);
    new_height = min(max(height*(1+height_scale_factor),1),image_sz(1)-1);
    
    new_x_temp = center_x+laplace_rand(5);
    new_center_x = min(image_sz(2)-new_width/2,max(new_width/2,new_x_temp));
    new_center_x = min(max(new_center_x,center_x-width),center_x+width);
    
    new_y_temp = center_y+laplace_rand(5);
    new_center_y = min(image_sz(1)-new_height/2,max(new_height/2,new_y_temp));
    new_center_y = min(max(new_center_y,center_y-height),center_y+height);
    
    bbox_curr_shift(1:4,i) = [new_center_x,new_center_y,new_center_x,new_center_y]-...
        [new_width,new_height,-new_width,-new_height]/2;
end
end %%function

function lp = laplace_rand(lambda)
u = rand(1)-0.5;
lp = sign(u)*log(1-abs(2*u))/lambda;
end %%function