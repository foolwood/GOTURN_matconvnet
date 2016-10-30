function [net, info] = train_goturn(varargin)
run vl_setupnn
addpath('../utils');
opts.dataDir = fullfile(pwd, '..', 'data') ;
opts.network = [] ;
opts.networkName = 'GOTURN';
opts.numFetchThreads = 12 ;%TODO
opts.version = 1+ismac();  % 1 [VOT+NUS_PRO] 2 [VOT+NUS_PRO]-lite 3 det16 4 NUS_PRO 5 full

opts.expDir = fullfile(pwd, '..', 'data', ['VOT+NUS_PRO-2-' opts.networkName]) ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

if ispc()
    trainOpts.gpus = [2] ;
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

image_prev = vl_imreadjpeg(imdb.images.target(batch));
image_curr = vl_imreadjpeg(imdb.images.search(batch));
bbox_prev = imdb.images.bboxs_prev(1,1,:,batch) ;
bbox_curr = imdb.images.bboxs_curr(1,1,:,batch) ;

% for i = 1:numel(image_prev)   %visual
%     subplot(2,numel(image_prev),i*2-1);
%     imshow(uint8(image_prev{i}));
%     rectangle('Position',bbox_2_rect(bbox_prev(:,:,:,i)));
%     subplot(2,numel(image_prev),i*2);
%     imshow(uint8(image_curr{i}));
%     rectangle('Position',bbox_2_rect(bbox_curr(:,:,:,i)));
% end

[images,targets,bboxs] =...
    make_all_examples(10,image_prev,image_curr,bbox_prev,bbox_curr,opts.sz);
images = bsxfun(@minus, images, imdb.images.data_mean);
targets = bsxfun(@minus, targets, imdb.images.data_mean);

if opts.numGpus > 0
    targets = gpuArray(targets) ;
    images = gpuArray(images) ;
    bboxs = gpuArray(bboxs) ;
end
inputs = {'target', targets,'image', images, 'bbox', bboxs} ;
end