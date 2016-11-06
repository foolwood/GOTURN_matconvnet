function [net, info] = train_goturn(varargin)
addpath('../utils');
run vl_setupnn;
opts.dataDir = fullfile('..', 'data') ;
opts.networkName = 'ALEX';
opts.numFetchThreads = 12 ;%TODO
opts.version = 10;
opts.expDir = fullfile('..', 'data', [opts.networkName '-experiment-' num2str(opts.version)]) ;
opts.imdbPath = fullfile('..', 'data',['GOTURN-experiment-' num2str(opts.version)], 'imdb.mat');

if ispc()
    trainOpts.gpus = [1];
else
    trainOpts.gpus = [];
end

trainOpts.learningRate = 1e-3;
trainOpts.weightDecay = 0.0005;
trainOpts.numEpochs = 50;
trainOpts.batchSize = 50;
opts.train = trainOpts;

if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

switch(opts.networkName)
    case 'GOTURN',
    net = goturn_net_init();
    case 'ALEX',
    net = alex_net_init();
    case 'VGG',
    net = vgg16_net_init();
    case 'ResNet',
    net = resnet50_net_init();
end

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
    targets = vl_imreadjpeg(imdb.images.target(batch),...
        'NumThreads',32,'Pack','Resize',opts.sz,'SubtractAverage', imdb.images.data_mean,'GPU');
    images = vl_imreadjpeg(imdb.images.image(batch),...
        'NumThreads',32,'Pack','Resize',opts.sz,'SubtractAverage', imdb.images.data_mean,'GPU');
    bboxs = gpuArray(single(imdb.images.bboxs(1,1,1:4,batch)));
else
    targets = vl_imreadjpeg(imdb.images.target(batch),...
        'NumThreads',32,'Pack','Resize',opts.sz,'SubtractAverage', imdb.images.data_mean);
    images = vl_imreadjpeg(imdb.images.image(batch),...
        'NumThreads',32,'Pack','Resize',opts.sz,'SubtractAverage', imdb.images.data_mean);
    bboxs = single(imdb.images.bboxs(1,1,1:4,batch));
end

inputs = {'target', targets{1},'image', images{1}, 'bbox', bboxs} ;
end