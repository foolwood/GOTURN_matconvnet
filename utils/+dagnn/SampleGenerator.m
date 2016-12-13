%DAGNN.SampleGenerator Generate sample for GOTURN
% input:
%     -bbox_prev_gt       1x4
%     -bbox_curr_gt       1x4
%     -image_prev      HxWx[3\1]x1
%     -image_curr      HxWx[3\1]x1
% output:
%     -bbox_gt_scaled          1x 1x4x(1+kGeneratedExamplesPerImage)
%     -image_target_pad       HoxWox3x(1+kGeneratedExamplesPerImage)
%     -image_search_pad       HoxWox3x(1+kGeneratedExamplesPerImage)
%   2016 Qiang Wang
classdef SampleGenerator < dagnn.Layer
    
    properties
        Ho = 0;
        Wo = 0;
        kGeneratedExamplesPerImage = 0;
        averageImage = reshape(single([123,117,104]),[1,1,3]);
        padding = 1;
        visual = true;
    end
    
    properties (Transient)
        % the grid --> this is cached
        % has the size: [2 x HoWo]
        yyxx ;
    end
    
    methods
        
        function outputs = forward(obj, inputs, ~)
            image_prev = inputs{1};
            image_curr = inputs{2};
            bbox_prev = inputs{3};
            bbox_curr = inputs{4};
            
            
            % generate the grid coordinates:
            useGPU = isa(bbox_prev, 'gpuArray');
            if isempty(obj.yyxx)
                obj.initGrid(useGPU);
            end
            
            [im_h,im_w,im_c,~] = size(image_prev);
            if im_c == 1
                image_prev = repmat(image_prev,[1,1,3,1]);
                image_curr = repmat(image_curr,[1,1,3,1]);
            end
            
            %% target
            roi_w = (1+obj.padding)*(bbox_prev(3)-bbox_prev(1));
            roi_h = (1+obj.padding)*(bbox_prev(4)-bbox_prev(2));
            roi_cx = (bbox_prev(1)+bbox_prev(3))/2;
            roi_cy = (bbox_prev(2)+bbox_prev(4))/2;
            
            cy_t = (roi_cy*2/(im_h-1))-1;
            cx_t = (roi_cx*2/(im_w-1))-1;
            
            h_s = roi_h/(im_h-1);
            w_s = roi_w/(im_w-1);
            
            s = [h_s;w_s]; % x,y scaling
            t = [cy_t;cx_t]; % translation
            
            g = bsxfun(@times, obj.yyxx, s); % scale
            g = bsxfun(@plus, g, t); % translate
            g = reshape(g, 2, obj.Ho, obj.Wo, 1);
            
            target_pad = vl_nnbilinearsampler(image_prev, g);
            targets = repmat(target_pad,[1,1,1,obj.kGeneratedExamplesPerImage+1]);
            
            if useGPU, %buff
                bboxes_gt_scaled = gpuArray.zeros([1,1,4,obj.kGeneratedExamplesPerImage+1],'single');%gpu support
                images = gpuArray.zeros(size(targets),'single');
            else
                bboxes_gt_scaled = zeros([1,1,4,obj.kGeneratedExamplesPerImage+1],'single');
                images = zeros(size(targets),'single');
            end
            images(:,:,:,1) = vl_nnbilinearsampler(image_curr, g);
            
            curr_search_location = [roi_cx-roi_w/2;roi_cy-roi_h/2];
            bbox_gt_recentered = recenter(bbox_curr',curr_search_location);
            bboxes_gt_scaled(1,1,1:4,1) = scale(bbox_gt_recentered,roi_w,roi_h);
            
            %% search
            if obj.kGeneratedExamplesPerImage > 0
                bboxes_curr_shift = shift([im_h,im_w],bbox_curr,obj.kGeneratedExamplesPerImage,useGPU);
                
                roi_w = (1+obj.padding)*(bboxes_curr_shift(3,:)-bboxes_curr_shift(1,:));
                roi_h = (1+obj.padding)*(bboxes_curr_shift(4,:)-bboxes_curr_shift(2,:));
                roi_cx = (bboxes_curr_shift(1,:)+bboxes_curr_shift(3,:))/2;
                roi_cy = (bboxes_curr_shift(2,:)+bboxes_curr_shift(4,:))/2;
                
                rand_search_location = [roi_cx-roi_w/2;roi_cy-roi_h/2];
                bboxes_gt_recentered = recenter(bbox_curr',rand_search_location);
                bboxes_gt_scaled(1,1,1:4,2:end) = scale(bboxes_gt_recentered,roi_w,roi_h);
                
                cy_t = (roi_cy*(2/(im_h-1)))-1;
                cx_t = (roi_cx*(2/(im_w-1)))-1;
                
                h_s = roi_h/(im_h-1);
                w_s = roi_w/(im_w-1);
                
                s = reshape([h_s;w_s],2,1,[]); % x,y scaling
                t = reshape([cy_t;cx_t],2,1,[]); % translation
                
                g = bsxfun(@times, obj.yyxx, s); % scale
                g = bsxfun(@plus, g, t); % translate
                g = reshape(g, 2,obj.Ho,obj.Wo,[]);
                
                images(:,:,:,2:end) = vl_nnbilinearsampler(image_curr, g);
            end
            
            if obj.visual
                scaled2rect = @(x,w) [x(1),x(2),x(3)-x(1),x(4)-x(2)]/10*(w-1)+1;
                
                for i = 1:(obj.kGeneratedExamplesPerImage+1)
                    subplot(4,ceil((obj.kGeneratedExamplesPerImage+1)/4),i);imshow(uint8(images(:,:,:,i)));
                    rectangle('Position',scaled2rect(gather(bboxes_gt_scaled(1,1,1:4,i)),obj.Wo),'EdgeColor',[0 1 0]);
                end
                drawnow;
            end
            
            targets = bsxfun(@minus,targets,obj.averageImage);
            images = bsxfun(@minus,images,obj.averageImage);
            outputs = {targets,images,bboxes_gt_scaled};
        end
        
        function obj = SampleGenerator(varargin)
            obj.load(varargin);
            % get the output sizes:
            obj.Ho = obj.Ho;
            obj.Wo = obj.Wo;
            obj.kGeneratedExamplesPerImage = obj.kGeneratedExamplesPerImage;
            obj.padding = obj.padding;
            obj.averageImage = obj.averageImage;
            obj.visual = obj.visual;
            obj.yyxx = [];
        end
        
        function obj = reset(obj)
            reset@dagnn.Layer(obj) ;
            obj.yyxx = [] ;
        end
        
        function initGrid(obj, useGPU)
            % initialize the grid:
            % this is a constant
            yi = linspace(-1, 1, obj.Ho);
            xi = linspace(-1, 1, obj.Wo);
            [xx,yy] = meshgrid(xi,yi);
            yyxx_ = single([yy(:), xx(:)]') ; % 2xM
            if useGPU
                yyxx_ = gpuArray(yyxx_);
            end
            obj.yyxx = yyxx_ ;
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
bbox_curr = gather(bbox_curr);
width = bbox_curr(3) - bbox_curr(1);
height = bbox_curr(4) - bbox_curr(2);
center_x = (bbox_curr(1) + bbox_curr(3))/2;
center_y = (bbox_curr(2) + bbox_curr(4))/2;

width_scale_factor = max(min(laplace_rand(15,n),0.4),-0.4);
new_width = min(max(width*(1+width_scale_factor),1),image_sz(2)-1);

height_scale_factor = max(min(laplace_rand(15,n),0.4),-0.4);
new_height = min(max(height*(1+height_scale_factor),1),image_sz(1)-1);

new_x_temp = center_x+width*max(min(laplace_rand(5,n),0.2),-0.2);
new_center_x = min(image_sz(2)-new_width/2,max(new_width/2,new_x_temp));
new_center_x = min(max(new_center_x,center_x-width),center_x+width);

new_y_temp = center_y+height*max(min(laplace_rand(5,n),0.2),-0.2);
new_center_y = min(image_sz(1)-new_height/2,max(new_height/2,new_y_temp));
new_center_y = min(max(new_center_y,center_y-height),center_y+height);

bbox_curr_shift(1,1:n) = new_center_x - new_width/2;
bbox_curr_shift(2,1:n) = new_center_y - new_height/2;
bbox_curr_shift(3,1:n) = new_center_x + new_width/2;
bbox_curr_shift(4,1:n) = new_center_y + new_height/2;

if useGPU
    bbox_curr_shift = gpuArray(bbox_curr_shift);
end

end %%function

function lp = laplace_rand(lambda,n)
u = rand(1,n,'single')-0.5;
lp = (sign(u).*log(1-abs(2*u)))/lambda;
end %%function