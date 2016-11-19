function [net, info] = train_goturn(varargin)
addpath('../utils');
run vl_setupnn;
opts.dataDir = fullfile('..', 'data') ;
opts.networkName = 'GOTURN_crop';
opts.numFetchThreads = 12 ;%TODO
opts.version = 1;
opts.expDir = fullfile('..', 'data', [opts.networkName '-experiment-' num2str(opts.version)]) ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

if ispc()
    trainOpts.gpus = [1];
else
    trainOpts.gpus = [];
end

trainOpts.learningRate = 1e-5;
trainOpts.weightDecay = 0.0005;
trainOpts.numEpochs = 20;
trainOpts.batchSize = 1;
opts.train = trainOpts;

if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

net = goturn_crop_net_init();

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
% imdb = setup_data_ram('dataDir', opts.dataDir,'version',opts.version);

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

[net, info] = cnn_train_dag(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    opts.train, ...
    'val', find(imdb.images.set == 2));

% -------------------------------------------------------------------------
%                                               (pc-1 mac-5 linux-3) Deploy
% -------------------------------------------------------------------------

net = goturn_deploy(net);

modelPath = fullfile(opts.expDir,...
    ['GOTURN-' num2str(ispc()*1+ismac()*2+isunix()*3),...
    '-Epochs' num2str(trainOpts.numEpochs) '.mat']);

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
    image_target = vl_imreadjpeg(imdb.images.target(batch),'NumThreads',32,'GPU');
    image_target = image_target{1};
    image_search = vl_imreadjpeg(imdb.images.search(batch),'NumThreads',32,'GPU');
    image_search = image_search{1};
%     image_target = gpuArray(imdb.images.target{batch});
%     image_search = gpuArray(imdb.images.search{batch});
    
    bbox_target = gpuArray(imdb.images.target_bboxs(1,1,1:4,batch));
    bbox_search = gpuArray(imdb.images.search_bboxs(1,1,1:4,batch));
else
    image_target = vl_imreadjpeg(imdb.images.target(batch),'NumThreads',32);
    image_target = image_target{1};
    image_search = vl_imreadjpeg(imdb.images.search(batch),'NumThreads',32);
    image_search = image_search{1};

%     image_target = imdb.images.target{batch};
%     image_search = imdb.images.search{batch};
    bbox_target = imdb.images.target_bboxs(1,1,1:4,batch);
    bbox_search = imdb.images.search_bboxs(1,1,1:4,batch);
end

inputs = {'bbox_target',bbox_target,'bbox_search',bbox_search,...
    'image_target',image_target,'image_search',image_search} ;
end