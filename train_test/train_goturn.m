function [net, info] = train_goturn(varargin)
addpath('../utils');
run vl_setupnn;
opts.dataDir = fullfile('..', 'data') ;
opts.numFetchThreads = 12 ;%TODO
opts.version = 1;
opts.expDir = fullfile('..', 'data', ['GOTURN-experiment-' num2str(opts.version)]) ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

trainOpts.learningRate = 1e-5;
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
    tic
    imdb = setup_data('dataDir', opts.dataDir,'version',opts.version);
    toc
    if ~exist(opts.expDir,'dir'),mkdir(opts.expDir);end
    save(opts.imdbPath, '-v7.3', '-struct', 'imdb');
end

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

[net, info] = cnn_train_dag(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    opts.train, ...
    'val', find(imdb.images.set == 2));

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
    image_target = vl_imreadjpeg(imdb.images.target(batch),'GPU');
    image_target = image_target{1};
    image_search = vl_imreadjpeg(imdb.images.search(batch),'GPU');
    image_search = image_search{1};
    bbox_target = gpuArray(imdb.images.target_bboxs(batch,1:4));
    bbox_search = gpuArray(imdb.images.search_bboxs(batch,1:4));
else
    image_target = vl_imreadjpeg(imdb.images.target(batch));
    image_target = image_target{1};
    image_search = vl_imreadjpeg(imdb.images.search(batch));
    image_search = image_search{1};
    bbox_target = imdb.images.target_bboxs(batch,1:4);
    bbox_search = imdb.images.search_bboxs(batch,1:4);
end

inputs = {'bbox_target',bbox_target,'bbox_search',bbox_search,...
    'image_target',image_target,'image_search',image_search} ;
end