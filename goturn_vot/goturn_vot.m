function goturn_vot
% ncc VOT integration example
% 
% This function is an example of tracker integration into the toolkit.
% The implemented tracker is a very simple NCC tracker that is also used as
% the baseline tracker for challenge entries.
%

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

vl_setupnn();
% Initialize the tracker
[state, ~] = goturn_initialize(imread(image), region);

while true

    % **********************************
    % VOT: Get next frame
    % **********************************
    [handle, image] = handle.frame(handle);

    if isempty(image)
        break;
    end;
    
	% Perform a tracking step, obtain new region
    [state, region] = goturn_update(state, imread(image));
    
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

function [state, location] = goturn_initialize(I, region, varargin)

    state.net = dagnn.DagNN.loadobj(load('GOTURN_net.mat'));
    state.net.mode = 'test';
    
    state.image_prev = I;
    state.bbox_prev_tight = region;
    state.bbox_prev_prior_tight = region;

    location = region;

end

function [state, location] = goturn_update(state, I, varargin)

    image_curr = I;
    
    target_pad = crop_pad_image(state.bbox_prev_tight,state.image_prev);
    [curr_search_region,search_location,edge_spacing_x,...
        edge_spacing_y] = crop_pad_image(state.bbox_prev_prior_tight,image_curr);
    
    bbox_estimate = regressor_regress(state.net,curr_search_region,target_pad);
    
    %%unscale the estimation to the real image size
    bbox_estimate_unscaled = bb_unscale(bbox_estimate,curr_search_region);
    %%find the estimated bounding box location relative to the current crop
    bbox_estimate_uncentered = bb_uncenter(bbox_estimate_unscaled,image_curr,...
        search_location,edge_spacing_x,edge_spacing_y);
    
    state.image_prev = I;
    state.bbox_prev_tight = bbox_estimate_uncentered;
    state.bbox_prev_prior_tight = bbox_estimate_uncentered;%TODO

    location = double(bbox_estimate_uncentered);

end
