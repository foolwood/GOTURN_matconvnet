function [net, info] = train_goturn(varargin)
run vl_setupnn

opts.dataDir = fullfile(pwd, '..', 'data') ;
opts.network = [] ;
opts.networkName = 'GOTURN';
opts.batchNormalization = true ;
opts.numFetchThreads = 12 ;

sfx = opts.networkName ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
opts.expDir = fullfile(pwd, '..', 'data', ['VOT-' sfx]) ;
if ispc()
    opts.imdbPath = 'E:\goturn_train\imdb.mat';
else
    opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
end


if ispc()
    trainOpts.gpus = [] ;
else
    trainOpts.gpus = [] ;
end

trainOpts.learningRate = 0.001 ;
trainOpts.numEpochs = 50 ;
trainOpts.batchSize = 30;
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
    imdb = vot_setup_data('dataDir', opts.dataDir) ;
    toc
    %mkdir(opts.expDir) ;
    %save(opts.imdbPath, '-v7.3', '-struct', 'imdb') ;
end

% imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
% if exist(imageStatsPath,'file')
%     load(imageStatsPath, 'averageImage', 'rgbMean') ;
% else
%     train = imdb.images.set == 1 ;
%     images = imdb.images(:,:,:,train) ;
%     [averageImage, rgbMean] = getImageStats(images,'imageSize', net) ;
%     save(imageStatsPath, 'averageImage', 'rgbMean') ;
% end



% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

[net, info] = cnn_train_dag(net, imdb, getBatch(opts), ...
                      'expDir', opts.expDir, ...
                      opts.train, ...
                      'val', find(imdb.images.set == 2)) ;

end

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
bopts = struct('numGpus', numel(opts.train.gpus)) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
targets = imdb.images.target(:,:,:,batch);
images = imdb.images.search(:,:,:,batch) ;
bboxs = imdb.images.bboxs(1,1,:,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'target', targets,'image', images, 'bbox', bboxs} ;
end