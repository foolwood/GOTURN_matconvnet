function [net, info] = train_goturn(varargin)
run vl_setupnn
addpath('../utils');
opts.dataDir = fullfile(pwd, '..', 'data') ;
opts.network = [] ;
opts.networkName = 'GOTURN';
opts.numFetchThreads = 12 ;%TODO
opts.version = 1;  % 1 [VOT+NUS_PRO] 2 NUS_PRO 3 det16
opts.expDir = fullfile(pwd, '..', 'data', ['VOT+NUS_PRO-' opts.networkName]) ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

if ispc()
    trainOpts.gpus = [] ;
else
    trainOpts.gpus = [] ;
end

trainOpts.learningRate = 1e-3 ;
trainOpts.weightDecay = 0.0005;
trainOpts.numEpochs = 50;
trainOpts.batchSize = 5;
opts.train = trainOpts;

if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

if isempty(opts.network)
    net = goturn_net_init() ;
else
    net = opts.network ;
    opts.network = [] ;
end

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath,'file')
    imdb = load(opts.imdbPath) ;
else
    tic
    imdb = vot_setup_data('dataDir', opts.dataDir,'version',opts.version) ;
    toc
    if ~exist(opts.expDir,'dir'),mkdir(opts.expDir) ;end
    save(opts.imdbPath, '-v7.3', '-struct', 'imdb') ;
end

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

[net, info] = cnn_train_dag(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    opts.train, ...
    'val', find(imdb.images.set == 2)) ;

% -------------------------------------------------------------------------
%                                               (pc-1 mac-5 linux-3) Deploy
% -------------------------------------------------------------------------

net = goturn_deploy(net) ;

modelPath = fullfile(opts.expDir,...
    ['GOTURN-' num2str(ispc()*1+ismac()*2+isunix()*3),...
    '-Epochs' num2str(trainOpts.numEpochs) '.mat']);

net_struct = net.saveobj() ;
save(modelPath, '-struct', 'net_struct') ;
clear net_struct ;


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
targets = vl_imreadjpeg(imdb.images.target(batch),...
    'NumThreads',20,'Pack','Resize',opts.sz,'SubtractAverage', imdb.images.data_mean);
targets = targets{1};
images = vl_imreadjpeg(imdb.images.search(batch),...
    'NumThreads',20,'Pack','Resize',opts.sz,'SubtractAverage', imdb.images.data_mean);
images = images{1};
bboxs = single(imdb.images.bboxs(1,1,1:4,batch));

if opts.numGpus > 0
    targets = gpuArray(targets);
    images = gpuArray(images);
    bboxs = gpuArray(bboxs) ; 
end

inputs = {'target', targets,'image', images, 'bbox', bboxs} ;
end