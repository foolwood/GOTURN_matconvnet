function goturn_vot
% goturn VOT

% *************************************************************
% VOT: Always call exit command at the end to terminate Matlab!
% *************************************************************
% cleanup = onCleanup(@() exit() );

% *************************************************************
% VOT: Set random seed to a different value every time.
% *************************************************************
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));

% **********************************
% VOT: Get initialization data
% **********************************
[handle, image, region] = vot('rectangle');

vl_setupnn();
% Initialize the tracker
[state, ~] = goturn_initialize(vl_imreadjpeg({image}), region);

while true

    % **********************************
    % VOT: Get next frame
    % **********************************
    [handle, image] = handle.frame(handle);

    if isempty(image)
        break;
    end;
    
	% Perform a tracking step, obtain new region
    [state, region] = goturn_update(state, vl_imreadjpeg({image}));
    
    % **********************************
    % VOT: Report position for frame
    % **********************************
    handle = handle.report(handle, region);
    
end;

% **********************************
% VOT: Output the results
% **********************************
% handle.quit(handle);

end

function [state, location] = goturn_initialize(I, region, varargin)

    state.visualization = true;
    
    state.net = dagnn.DagNN.loadobj(load('GOTURN_trained.mat'));
    state.net.layers(1).block.No = 1;
   
%     state.net.mode = 'test';
    state.rect2bbox = @(x) ([x(1)-1,x(2)-1,x(1)-1+x(3),x(2)-1+x(4)]);
    state.bbox2rect = @(x) ([x(1)+1,x(2)+1,x(3)-x(1),x(4)-x(2)]);
    
    state.image_prev = I{1};
    state.sz = size(I{1});
    state.rect_prev_tight = single(region);
    state.rect_prev_prior_tight = single(region);
    
   
    location = region;
    if state.visualization
        imshow(uint8(I{1}));
        rectangle('Position',location,'EdgeColor',[0,1,0]);
        drawnow;
    end
    
end

function [state, location] = goturn_update(state, I, varargin)

    image_curr = I{1};
    bbox_prev_tight = state.rect2bbox(state.rect_prev_tight);
    bbox_prev_prior_tight = state.rect2bbox(state.rect_prev_prior_tight);
    
    state.net.eval({'bbox_target',bbox_prev_tight,...
        'bbox_search',bbox_prev_prior_tight,...
        'image_target',state.image_prev,'image_search',image_curr});
    
    bbox_estimate = squeeze(state.net.vars(state.net.getVarIndex('fc8')).value)';
    crop_width = 2*state.rect_prev_tight(3);
    crop_height = 2*state.rect_prev_tight(4);
    bbox_estimate_unscaled = bbox_estimate/10.*...
        [crop_width,crop_height,crop_width,crop_height];
    bbox_estimate_uncentered = max(0,bbox_estimate_unscaled+...
        bbox_prev_prior_tight);
    
    bbox_estimate_uncentered(3) = min(state.sz(2)-1,bbox_estimate_uncentered(3));
    bbox_estimate_uncentered(4) = min(state.sz(1)-1,bbox_estimate_uncentered(4));
    rect_estimate_uncentered = state.bbox2rect(bbox_estimate_uncentered);
    
    state.image_prev = image_curr;
    state.rect_prev_tight = single(rect_estimate_uncentered);
    state.rect_prev_prior_tight = single(rect_estimate_uncentered);%TODO

    location = double(rect_estimate_uncentered);
    
    if state.visualization
        imshow(uint8(image_curr));
        rectangle('Position',location,'EdgeColor',[0,1,0]);
        drawnow;
    end

end
