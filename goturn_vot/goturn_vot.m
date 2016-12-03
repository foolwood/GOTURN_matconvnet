function goturn_vot
% goturn VOT

% *************************************************************
% VOT: Always call exit command at the end to terminate Matlab!
% *************************************************************
cleanup = onCleanup(@() exit() );

% *************************************************************
% VOT: Set random seed to a different value every time.
% *************************************************************
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));

% **********************************
% VOT: Get initialization data
% **********************************
[handle, image, region] = vot('rectangle');

addpath('../utils');
vl_setupnn();
% Initialize the tracker
[state, ~] = goturn_initialize(image, region);

while true
    
    % **********************************
    % VOT: Get next frame
    % **********************************
    [handle, image] = handle.frame(handle);
    
    if isempty(image)
        break;
    end;
    
    % Perform a tracking step, obtain new region
    [state, region] = goturn_update(state, image);
    
    % **********************************
    % VOT: Report position for frame
    % **********************************
    handle = handle.report(handle, region);
    
end;

% **********************************
% VOT: Output the results
% **********************************
handle.quit(handle);

end

function [state, location] = goturn_initialize(image, region, varargin)

state = [];
state.gpu = ispc();
state.visualization = true;

state.net = dagnn.DagNN.loadobj(load('GOTURN_trained.mat'));
% state.net.mode = 'test';
state.net.layers(1).block.No = 1;

if state.gpu,state.net.move('gpu');end;
if state.visualization
    state.net.vars(state.net.getVarIndex('image_target_crop')).precious = 1;
    state.net.vars(state.net.getVarIndex('image_search_crop')).precious = 1;
end

state.rect2bbox = @(x) ([x(1)-1,x(2)-1,x(1)-1+x(3),x(2)-1+x(4)]);
state.bbox2rect = @(x) ([x(1)+1,x(2)+1,x(3)-x(1),x(4)-x(2)]);

if state.gpu
    state.image_prev = vl_imreadjpeg({image},'GPU');
else
    state.image_prev = vl_imreadjpeg({image});
end

state.sz = size(state.image_prev{1});
state.rect_prev_tight = single(region);
state.rect_prev_prior_tight = single(region);

location = region;

if state.visualization
    figure;
    subplot(1,3,1);state.h1 = imshow(uint8(state.image_prev{1}));
    state.h2 = rectangle('Position',location,'EdgeColor',[0,1,0]);
    subplot(1,3,2);state.h3 = imshow(zeros(227,227,3,'single'));
    subplot(1,3,3);state.h4 = imshow(zeros(227,227,3,'single'));
    state.h5 = rectangle('Position',[1,1,1,1],'EdgeColor',[0,1,0]);
    drawnow;
end

end

function [state, location] = goturn_update(state, image, varargin)

if state.gpu
    image_curr = vl_imreadjpeg({image},'GPU');
else
    image_curr = vl_imreadjpeg({image});
end

bbox_prev_tight = state.rect2bbox(state.rect_prev_tight);
bbox_prev_prior_tight = state.rect2bbox(state.rect_prev_prior_tight);

if state.gpu 
    state.net.eval({'bbox_target',gpuArray(bbox_prev_tight),...
    'bbox_search',gpuArray(bbox_prev_prior_tight),...
    'image_target',state.image_prev{1},'image_search',image_curr{1}});
else
    state.net.eval({'bbox_target',bbox_prev_tight,...
    'bbox_search',bbox_prev_prior_tight,...
    'image_target',state.image_prev{1},'image_search',image_curr{1}});
end

bbox_estimate = gather(squeeze(state.net.vars(state.net.getVarIndex('fc8')).value))';

crop_width = 2*state.rect_prev_tight(3);
crop_height = 2*state.rect_prev_tight(4);
bbox_estimate_unscaled = bbox_estimate/10.*...
    [crop_width,crop_height,crop_width,crop_height];

rt_crop = (bbox_prev_tight([1,2])+bbox_prev_tight([3,4]))/2-...
    [crop_width,crop_height]/2;

bbox_estimate_uncentered = max(0,bbox_estimate_unscaled+...
    rt_crop([1,2,1,2]));

bbox_estimate_uncentered(3) = min(state.sz(2)-1,bbox_estimate_uncentered(3));
bbox_estimate_uncentered(4) = min(state.sz(1)-1,bbox_estimate_uncentered(4));
rect_estimate_uncentered = state.bbox2rect(bbox_estimate_uncentered);

state.image_prev = image_curr;
state.rect_prev_tight = single(rect_estimate_uncentered);
state.rect_prev_prior_tight = single(rect_estimate_uncentered);%TODO

location = double(rect_estimate_uncentered);

if state.visualization
    scaled2rect = @(x) [x(1),x(2),x(3)-x(1),x(4)-x(2)]/10*227+1;
    
    state.h1.set('CData',uint8(state.image_prev{1}));
    state.h2.set('Position',location);
    
    image_prev_crop = bsxfun(@plus,...
        state.net.vars(state.net.getVarIndex('image_target_crop')).value,...
        state.net.meta.normalization.averageImage);
    
    image_curr_crop = bsxfun(@plus,...
        state.net.vars(state.net.getVarIndex('image_search_crop')).value,...
        state.net.meta.normalization.averageImage);
    
    state.h3.set('CData',uint8(gather(image_prev_crop)));
    state.h4.set('CData',uint8(gather(image_curr_crop)));
    state.h5.set('Position',scaled2rect(bbox_estimate));
    drawnow;
end

end
