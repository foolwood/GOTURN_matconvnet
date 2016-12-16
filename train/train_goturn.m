function [net, info] = train_goturn(varargin)
addpath('../utils');
run vl_setupnn;
opts.dataDir = fullfile('..', 'data') ;
opts.version = 1;
opts.expDir = fullfile('..', 'data', ['GOTURN-experiment-' num2str(opts.version)]) ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

trainOpts.learningRate = [1e-4*ones(1,10),10e-5*ones(1,10)];
trainOpts.weightDecay = 0.0005;
trainOpts.numEpochs = 20;
trainOpts.batchSize = 1;
opts.train = trainOpts;

if ~isfield(opts.train, 'gpus') && ispc(), opts.train.gpus = [2];
else opts.train.gpus = [];end;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

net = goturn_net_init();

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath,'file')
    imdb = load(opts.imdbPath) ;
else
    imdb = setup_data('version',opts.version);
    if ~exist(opts.expDir,'dir'),mkdir(opts.expDir);end
    save(opts.imdbPath, '-v7.3', '-struct', 'imdb');
end

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

[net, info] = cnn_train_dag(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, opts.train);

% -------------------------------------------------------------------------
%                                                                   Deploy
% -------------------------------------------------------------------------

net = goturn_deploy(net);

modelPath = fullfile(opts.expDir,'GOTURN_trained.mat');

net_struct = net.saveobj();
save(modelPath, '-struct', 'net_struct');
clear net_struct;

end

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
bopts = struct('numGpus', numel(opts.train.gpus),'sz',[227,227]) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
if opts.numGpus > 0
    if imdb.images.set(batch) == 2 || mod(randi(2,1),imdb.type) == 0
        image_prev = vl_imreadjpeg(imdb.images.video_images_prev(batch),'GPU');
        image_prev = image_prev{1};
        image_curr = vl_imreadjpeg(imdb.images.video_images_curr(batch),'GPU');
        image_curr = image_curr{1};
        bbox_prev = gpuArray(imdb.images.video_bbox_prev(batch,1:4));
        bbox_curr = gpuArray(imdb.images.video_bbox_curr(batch,1:4));
    else
        batch = randi(imdb.images.image_n_valid,1);
        image_prev = vl_imreadjpeg(imdb.images.image_path(batch),'GPU');
        image_prev = image_prev{1};
        image_curr = image_prev;
        image_sz = size(image_curr);
        image_sz = image_sz([2,1]); %width height
        if(any(image_sz ~= imdb.images.image_display_sz(batch,:)))
            factor = image_sz./imdb.images.image_display_sz(batch,:);
            factor = factor([1,2,1,2]);
        else
            factor = [1,1,1,1];
        end
        
        bbox_prev = gpuArray(imdb.images.image_bbox(batch,:).*factor);
        bbox_curr = bbox_prev;
    end
else
    if imdb.images.set(batch) == 2 || mod(randi(2,1),imdb.type) == 0
        image_prev = vl_imreadjpeg(imdb.images.video_images_prev(batch));
        image_prev = image_prev{1};
        image_curr = vl_imreadjpeg(imdb.images.video_images_curr(batch));
        image_curr = image_curr{1};
        bbox_prev = imdb.images.video_bbox_prev(batch,1:4);
        bbox_curr = imdb.images.video_bbox_curr(batch,1:4);
    else
        batch = randi(imdb.images.image_n_valid,1);
        image_prev = vl_imreadjpeg(imdb.images.image_path(batch));
        image_prev = image_prev{1};
        image_curr = image_prev;
        image_sz = size(image_curr);
        image_sz = image_sz([2,1]); %width height
        if(any(image_sz ~= imdb.images.image_display_sz(batch,:)))
            factor = image_sz./imdb.images.image_display_sz(batch,:);
            factor = factor([1,2,1,2]);
        else
            factor = [1,1,1,1];
        end
        
        bbox_prev = imdb.images.image_bbox(batch,:).*factor;
        bbox_curr = bbox_prev;
    end
end

inputs = {'image_prev',image_prev,'image_curr',image_curr,...
    'bbox_prev',bbox_prev,'bbox_curr',bbox_curr} ;
end