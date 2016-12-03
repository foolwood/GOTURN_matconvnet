function [net, info] = train_goturn(varargin)
addpath('../utils');
run vl_setupnn;
opts.dataDir = fullfile('..', 'data') ;
opts.version = 2;
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
imdb.flag = imdb.flag+1;
if opts.numGpus > 0
    if mod(imdb.flag,imdb.type) == 0
        batch = randi(numel(imdb.images.video_target),1);
        image_target = vl_imreadjpeg(imdb.images.video_target(batch),'GPU');
        image_target = image_target{1};
        image_search = vl_imreadjpeg(imdb.images.video_search(batch),'GPU');
        image_search = image_search{1};
        bbox_target = gpuArray(imdb.images.video_target_bboxs(batch,1:4));
        bbox_search = gpuArray(imdb.images.video_search_bboxs(batch,1:4));
    else
        
        batch = randi(numel(imdb.images.image_path),1);
        while numel(imdb.images.image_bboxs(batch)) == 0
            batch = randi(numel(imdb.images.image_target),1);
        end
        image_target = vl_imreadjpeg(imdb.images.image_path(batch),'GPU');
        image_target = image_target{1};
        image_search = image_target;
       
        image_bboxs_temple = imdb.images.image_bboxs{batch};
        batch2 = randi(size(image_bboxs_temple,1),1);
        bbox_target = gpuArray(image_bboxs_temple(batch2,1:4));
        bbox_search = bbox_target;
    end
else
    if mod(imdb.flag,imdb.type) == 0
        batch = randi(numel(imdb.images.video_target),1);
        image_target = vl_imreadjpeg(imdb.images.video_target(batch));
        image_target = image_target{1};
        image_search = vl_imreadjpeg(imdb.images.video_search(batch));
        image_search = image_search{1};
        bbox_target = gpuArray(imdb.images.video_target_bboxs(batch,1:4));
        bbox_search = gpuArray(imdb.images.video_search_bboxs(batch,1:4));
    else
        batch = randi(numel(imdb.images.image_path),1);
        while numel(imdb.images.image_bboxs(batch)) == 0
            batch = randi(numel(imdb.images.image_path),1);
        end
        image_target = vl_imreadjpeg(imdb.images.image_target(batch),'GPU');
        image_target = image_target{1};
        image_search = image_target;
        
        image_bboxs_temple = imdb.images.image_bboxs{batch};
        batch2 = randi(size(image_bboxs_temple,1),1);
        bbox_target = image_bboxs_temple(batch2,1:4);
        bbox_search = bbox_target;
    end
end

inputs = {'bbox_target',bbox_target,'bbox_search',bbox_search,...
    'image_target',image_target,'image_search',image_search} ;
end