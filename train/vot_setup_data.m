function imdb = vot_setup_data(varargin)
rng('default');
addpath('../utils');
% opts = vl_argparse(opts, varargin) ;
nsample= 1+ispc()*49;
opts = [];
opts.dataDir = '../data';
opts.expDir = '../data/crop';
opts.version = 1;    % 1 [VOT+NUS_PRO] 2 NUS_PRO 3 det16
opts.size = [227,227];
opts = vl_argparse(opts, varargin) ;
opts.vot15_dataDir = fullfile(opts.dataDir,'VOT15');
opts.vot14_dataDir = fullfile(opts.dataDir,'VOT14');
opts.det16_dataDir = fullfile(opts.dataDir,'DET16');
opts.nus_pro_dataDir = fullfile(opts.dataDir,'NUS_PRO');
opts.visualization = false;
imdb = [];
% -------------------------------------------------------------------------
%                                                                 Data Init
% -------------------------------------------------------------------------

switch opts.version%vot15:21395 vot14:10188
    case 1,
        set_name = {'vot15','vot14','nus_pro'};
        set = [ones(1,21395*nsample) 2*ones(1,10188*nsample) ones(1,26090*nsample)];
        imdb.images.set = set;
        imdb.images.target = cell([numel(set),1]);%TODO:use name
        imdb.images.search = cell([numel(set),1]);%TODO:use name
        imdb.images.bboxs = zeros(1,1,4,numel(set),'single');
    case 2,
        set_name = {'nus_pro'};
        set = [ones(1,26090*nsample) 2*ones(1,0*nsample)];
        imdb.images.set = set;
        imdb.images.target = cell([numel(set),1]);%TODO:use name
        imdb.images.search = cell([numel(set),1]);%TODO:use name
        imdb.images.bboxs = zeros(1,1,4,numel(set),'single');
    case 3,
        set_name = {'det16'};
        set = [ones(1,456567) 2*ones(1,60000)];
        imdb.images.set = set;
        imdb.images.target = cell([numel(set),1]);%TODO:use name
        imdb.images.search = cell([numel(set),1]);%TODO:use name
        imdb.images.bboxs = zeros(1,1,4,numel(set),'single');
    case 4,
        
    otherwise,
        
end


now_index = 0;
expDir = opts.expDir;
% -------------------------------------------------------------------------
%                                                           VOT15
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'vot15'))
    
    disp('VOT2015 Data(for Training):');
    vot15_dataDir = opts.vot15_dataDir;
    dirs = dir(vot15_dataDir);
    videos = {dirs.name};
    videos(strcmp('.', videos) | strcmp('..', videos)| ~[dirs.isdir]) = [];
    
    parfor  v = 1:numel(videos)
        video = videos{v};disp(['      ' video]);
        [img_files, ground_truth_4xy] = load_video_info_vot(vot15_dataDir, video);
        bbox_gt = get_axis_aligned_BB(ground_truth_4xy);
        im_bank = vl_imreadjpeg(img_files);
        video_expDir = [expDir '/vot15/' video];
        if ~exist(video_expDir,'dir'),mkdir(video_expDir) ;end;
        for frame = 1:(numel(im_bank)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            make_all_examples(im_bank{frame},im_bank{frame+1},...
                bbox_gt(frame,:),bbox_gt(frame+1,:),nsample,video_frame_expDir);
        end %%end frame
    end %%end v
    
    
    for  v = 1:numel(videos)
        video = videos{v};disp(['      ' video]);
        [img_files, ~] = load_video_info_vot(vot15_dataDir, video);
        video_expDir = [expDir '/vot15/' video];
        for frame = 1:(numel(img_files)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            load([sprintf(video_frame_expDir,0,0),'.mat']);
            imdb.images.bboxs(1,1,1:4,now_index+(1:nsample)) = bbox_gt_scaled;
            for i = 1:nsample
                imdb.images.target(now_index+i) = {[sprintf(video_frame_expDir,0,i),'.jpg']};
                imdb.images.image(now_index+i) = {[sprintf(video_frame_expDir,1,i),'.jpg']};
            end
            now_index = now_index+nsample;
        end %%end frame
    end %%end v
end %%end vot15


% -------------------------------------------------------------------------
%                                                           VOT14
% -------------------------------------------------------------------------

if any(strcmpi(set_name,'vot14'))
    
    disp('VOT2014 Data(for Validation):');
    vot14_dataDir = opts.vot14_dataDir;
    dirs = dir(vot14_dataDir);
    videos = {dirs.name};
    videos(strcmp('.', videos) | strcmp('..', videos)| ~[dirs.isdir]) = [];
    
    parfor  v = 1:numel(videos)
        video = videos{v};disp(['      ' video]);
        [img_files, ground_truth_4xy] = load_video_info_vot(vot14_dataDir, video);
        bbox_gt = get_axis_aligned_BB(ground_truth_4xy);
        im_bank = vl_imreadjpeg(img_files);
        video_expDir = [expDir '/vot14/' video];
        if ~exist(video_expDir,'dir'),mkdir(video_expDir) ;end;
        for frame = 1:(numel(im_bank)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            make_all_examples(im_bank{frame},im_bank{frame+1},...
                bbox_gt(frame,:),bbox_gt(frame+1,:),nsample,video_frame_expDir);
        end %%end frame
    end %%end v
    
    
    for  v = 1:numel(videos)
        video = videos{v};disp(['      ' video]);
        [img_files, ~] = load_video_info_vot(vot14_dataDir, video);
        video_expDir = [expDir '/vot14/' video];
        for frame = 1:(numel(img_files)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            load([sprintf(video_frame_expDir,0,0),'.mat']);
            imdb.images.bboxs(1,1,1:4,now_index+(1:nsample)) = bbox_gt_scaled;
            for i = 1:nsample
                imdb.images.target(now_index+i) = {[sprintf(video_frame_expDir,0,i),'.jpg']};
                imdb.images.image(now_index+i) = {[sprintf(video_frame_expDir,1,i),'.jpg']};
            end
            now_index = now_index+nsample;
        end %%end frame
    end %%end v
end %%end vot14


% -------------------------------------------------------------------------
%                                                           NUS_PRO
% -------------------------------------------------------------------------

if any(strcmpi(set_name,'nus_pro'))
    
    disp('NUS_PRO Data(for Training):');
    nus_pro_dataDir = opts.nus_pro_dataDir;
    filename = fullfile(nus_pro_dataDir,'seq_list_with_gt.csv');
    videos = importdata(filename);
    
    parfor  v = 1:numel(videos)
        video = videos{v};disp(['      ' video]);
        [img_files, ground_truth_4xy] = load_video_info_nus_pro(nus_pro_dataDir, video);
        bbox_gt = get_axis_aligned_BB(ground_truth_4xy);
        im_bank = vl_imreadjpeg(img_files);
        video_expDir = [expDir '/nus_pro/' video];
        if ~exist(video_expDir,'dir'),mkdir(video_expDir) ;end;
        for frame = 1:(numel(im_bank)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            make_all_examples(im_bank{frame},im_bank{frame+1},...
                bbox_gt(frame,:),bbox_gt(frame+1,:),nsample,video_frame_expDir);
        end %%end frame
    end %%end v
    
    for  v = 1:numel(videos)
        video = videos{v};disp(['      ' video]);
        [img_files, ~] = load_video_info_nus_pro(nus_pro_dataDir, video);
        video_expDir = [expDir '/nus_pro/' video];
        for frame = 1:(numel(img_files)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            load([sprintf(video_frame_expDir,0,0),'.mat']);
            imdb.images.bboxs(1,1,1:4,now_index+(1:nsample)) = bbox_gt_scaled;
            for i = 1:nsample
                imdb.images.target(now_index+i) = {[sprintf(video_frame_expDir,0,i),'.jpg']};
                imdb.images.image(now_index+i) = {[sprintf(video_frame_expDir,i,i),'.jpg']};
            end
            now_index = now_index+nsample;
        end %%end frame
    end %%end v
end %%end nus-pro

dataMean(1,1,1:3) = single([123,117,104]);
imdb.images.data_mean(1,1,1:3) = dataMean;
imdb.images.size = opts.size;
imdb.meta.sets = {'train', 'val'} ;
end %%end function
