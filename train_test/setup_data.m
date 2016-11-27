function imdb = setup_data(varargin)
addpath('../utils');
addpath('../data/ILSVRC/devkit/evaluation');
opts = [];
opts.dataDir = '../data';
opts.version = 4;
opts = vl_argparse(opts, varargin) ;

opts.vot16_dataDir = fullfile(opts.dataDir,'VOT16');
opts.otb_dataDir = fullfile(opts.dataDir,'OTB');
opts.nus_pro_dataDir = fullfile(opts.dataDir,'NUS_PRO');
opts.tc128_dataDir = fullfile(opts.dataDir,'TC128');
opts.alov300_dataDir = fullfile(opts.dataDir,'ALOV300');
opts.det16_dataDir = fullfile(opts.dataDir,'ILSVRC');

opts.visualization = false;
imdb = [];

% -------------------------------------------------------------------------
%   full dataset:
%           vot16:21395 vot15:21395 vot14:10188 vot13:5665
%           cvpr2013:29435 tb100:58935 tb50:26922
%           nus_pro:26090 tc128:55217 alov300:89351 det16:478806
%   Special dataset:
%           alov300_goturn:15570
% -------------------------------------------------------------------------

switch opts.version
    case 1,
        bbox_mode = 'minmax';%
        set_name = {'vot16'};
        set = ones(1,21395);
        set(randperm(21395,1000)) = 2;
    case 2,
        bbox_mode = 'minmax';%
        set_name = {'nus_pro','tc128_no_cvpr2013','alov300','det16'};
        set = ones(1,26090+30507+89351+478806);
        set(randperm(numel(set),1000)) = 2;
    case 4,
        bbox_mode = 'minmax';%
        set_name = {'alov300_goturn','det16'};
        set = ones(1,15570+478806);
        set(randperm(numel(set),1000)) = 2;
    otherwise,
        error('No such version!'); 
end


if strcmp(bbox_mode,'axis_aligned'),
    get_bbox = @get_axis_aligned_BB;
else
    get_bbox = @get_minmax_BB;
end

imdb.images.set = set;
imdb.images.target = cell([numel(set),1]);
imdb.images.search = cell([numel(set),1]);
imdb.images.target_bboxs = zeros(numel(set),4,'single');
imdb.images.search_bboxs = zeros(numel(set),4,'single');

now_index = 0;
% -------------------------------------------------------------------------
%                                                                     VOT16
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'vot16'))
    
    disp('VOT2016 Data:');
    vot16_dataDir = opts.vot16_dataDir;
    videos = importdata(fullfile(vot16_dataDir,'list.txt'));
    
    for v = 1:numel(videos)
        video = videos{v};fprintf('%3d :%20s\n',v,video);
        [img_files, ground_truth_4xy] = load_video_info_vot(vot16_dataDir, video);
        bbox_gt = get_bbox(ground_truth_4xy);
        n_len = numel(img_files);
        imdb.images.target(now_index+(1:(n_len-1))) = img_files(1:(n_len-1));
        imdb.images.search(now_index+(1:(n_len-1))) = img_files(2:n_len);
        imdb.images.target_bboxs(now_index+(1:(n_len-1)),:) = bbox_gt(1:(n_len-1),:);
        imdb.images.search_bboxs(now_index+(1:(n_len-1)),:) = bbox_gt(2:n_len,:);
        now_index = now_index+(n_len-1);
    end %%end v
end %%end vot16

% -------------------------------------------------------------------------
%                                                                   NUS_PRO
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'nus_pro'))
    
    disp('NUS_PRO Data:');
    nus_pro_dataDir = opts.nus_pro_dataDir;
    filename = fullfile(nus_pro_dataDir,'seq_list_with_gt.csv');
    videos = importdata(filename);
    
    for  v = 1:numel(videos)
        video = videos{v};fprintf('%3d :%20s\n',v,video);
        [img_files, ground_truth_4xy] = load_video_info_nus_pro(nus_pro_dataDir, video);
        bbox_gt = get_bbox(ground_truth_4xy);
        n_len = numel(img_files);
        imdb.images.target(now_index+(1:(n_len-1))) = img_files(1:(n_len-1));
        imdb.images.search(now_index+(1:(n_len-1))) = img_files(2:n_len);
        imdb.images.target_bboxs(now_index+(1:(n_len-1)),:) = bbox_gt(1:(n_len-1),:);
        imdb.images.search_bboxs(now_index+(1:(n_len-1)),:) = bbox_gt(2:n_len,:);
        now_index = now_index+(n_len-1);
    end %%end v
end %%end nus-pro
% -------------------------------------------------------------------------
%                                                                     TC128
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'tc128'))
    
    disp('TC128 Data:');
    tc128_dataDir = opts.tc128_dataDir;
    TC128_temp = dir(tc128_dataDir);
    TC128 = {TC128_temp.name};
    TC128(strcmp('.', TC128) | strcmp('..', TC128)| ~[TC128_temp.isdir]) = [];
    videos = TC128;
    
    for  v = 1:numel(videos)
        video = videos{v};fprintf('%3d :%20s\n',v,video);
        [img_files, ground_truth_4xy] = load_video_info_tc128(tc128_dataDir, video);
        bbox_gt = get_bbox(ground_truth_4xy);
        n_len = numel(img_files);
        imdb.images.target(now_index+(1:(n_len-1))) = img_files(1:(n_len-1));
        imdb.images.search(now_index+(1:(n_len-1))) = img_files(2:n_len);
        imdb.images.target_bboxs(now_index+(1:(n_len-1)),:) = bbox_gt(1:(n_len-1),:);
        imdb.images.search_bboxs(now_index+(1:(n_len-1)),:) = bbox_gt(2:n_len,:);
        now_index = now_index+(n_len-1);
    end %%end v
end %%end tc128

% -------------------------------------------------------------------------
%                                                                   ALOV300
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'alov300'))
    
    disp('ALOV300 Data:');
    alov300_dataDir = opts.alov300_dataDir;
    ALOV300_temp1 = dir(fullfile(alov300_dataDir,'imagedata++'));
    ALOV300_temp2 = {ALOV300_temp1.name};
    ALOV300_temp2(strcmp('.', ALOV300_temp2) | strcmp('..', ALOV300_temp2)| ~[ALOV300_temp1.isdir]) = [];
    ALOV300 = cell(0);
    for i = 1:numel(ALOV300_temp2)
        ALOV300_temp3 = dir(fullfile(alov300_dataDir,'imagedata++',ALOV300_temp2{i}));
        ALOV300_temp4 = {ALOV300_temp3.name};
        ALOV300_temp4(strcmp('.', ALOV300_temp4) | strcmp('..', ALOV300_temp4)| ~[ALOV300_temp3.isdir]) = [];
        ALOV300(end+(1:numel(ALOV300_temp4))) = ALOV300_temp4;
    end
    videos = ALOV300;
    
    for  v = 1:numel(videos)
        video = videos{v};fprintf('%3d :%30s\n',v,video);
        [img_files, ground_truth_4xy] = load_video_info_alov300(alov300_dataDir, video);
        bbox_gt = get_bbox(ground_truth_4xy);
        n_len = numel(img_files);
        imdb.images.target(now_index+(1:(n_len-1))) = img_files(1:(n_len-1));
        imdb.images.search(now_index+(1:(n_len-1))) = img_files(2:n_len);
        imdb.images.target_bboxs(now_index+(1:(n_len-1)),:) = bbox_gt(1:(n_len-1),:);
        imdb.images.search_bboxs(now_index+(1:(n_len-1)),:) = bbox_gt(2:n_len,:);
        now_index = now_index+(n_len-1);
    end %%end v
end %%end alov300

% -------------------------------------------------------------------------
%                                                         tc128_no_cvpr2013
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'tc128_no_cvpr2013'))
    
    disp('TC128 Data(no cvpr2013 for Training):');
    tc128_dataDir = opts.tc128_dataDir;
    TC128_temp = dir(tc128_dataDir);
    TC128 = {TC128_temp.name};
    TC128(strcmp('.', TC128) | strcmp('..', TC128)| ~[TC128_temp.isdir]) = [];
    tc128_no_cvpr2013_index = zeros(1,numel(TC128));
    for k = 1:numel(TC128)
        if numel(strfind(TC128{k},'_ce'))~=0
            tc128_no_cvpr2013_index(k) = 1;
        else
            tc128_no_cvpr2013_index(k) = 0;
        end
    end
    videos = TC128(tc128_no_cvpr2013_index==1);
    
    for  v = 1:numel(videos)
        video = videos{v};fprintf('%3d :%20s\n',v,video);
        [img_files, ground_truth_4xy] = load_video_info_tc128(tc128_dataDir, video);
        bbox_gt = get_bbox(ground_truth_4xy);
        n_len = numel(img_files);
        imdb.images.target(now_index+(1:(n_len-1))) = img_files(1:(n_len-1));
        imdb.images.search(now_index+(1:(n_len-1))) = img_files(2:n_len);
        imdb.images.target_bboxs(now_index+(1:(n_len-1)),:) = bbox_gt(1:(n_len-1),:);
        imdb.images.search_bboxs(now_index+(1:(n_len-1)),:) = bbox_gt(2:n_len,:);
        now_index = now_index+(n_len-1);
    end %%end v
end %%end tc128_no_cvpr2013




% -------------------------------------------------------------------------
%                                                            ALOV300_GOTURN
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'alov300_goturn'))
    
    disp('ALOV300 Data:');
    alov300_dataDir = opts.alov300_dataDir;
    ALOV300_temp1 = dir(fullfile(alov300_dataDir,'imagedata++'));
    ALOV300_temp2 = {ALOV300_temp1.name};
    ALOV300_temp2(strcmp('.', ALOV300_temp2) | strcmp('..', ALOV300_temp2)| ~[ALOV300_temp1.isdir]) = [];
    ALOV300 = cell(0);
    for i = 1:numel(ALOV300_temp2)
        ALOV300_temp3 = dir(fullfile(alov300_dataDir,'imagedata++',ALOV300_temp2{i}));
        ALOV300_temp4 = {ALOV300_temp3.name};
        ALOV300_temp4(strcmp('.', ALOV300_temp4) | strcmp('..', ALOV300_temp4)| ~[ALOV300_temp3.isdir]) = [];
        ALOV300(end+(1:numel(ALOV300_temp4))) = ALOV300_temp4;
    end
    videos = ALOV300;
    videos_removed = {'01-Light_video00016','01-Light_video00022',...
        '01-Light_video00023','02-SurfaceCover_video00012',...
        '03-Specularity_video00003','03-Specularity_video00012',...
        '10-LowContrast_video00013'};
    for i = 1:numel(videos_removed)
        videos(strcmpi(videos,videos_removed{i})) = [];
    end
    
    for  v = 1:numel(videos)
        video = videos{v};fprintf('%3d :%30s\n',v,video);
        [img_files, ground_truth_4xy] = ...
            load_video_info_alov300(alov300_dataDir, video,false);
        bbox_gt = get_bbox(ground_truth_4xy);
        n_len = numel(img_files);
        imdb.images.target(now_index+(1:(n_len-1))) = img_files(1:(n_len-1));
        imdb.images.search(now_index+(1:(n_len-1))) = img_files(2:n_len);
        imdb.images.target_bboxs(now_index+(1:(n_len-1)),:) = bbox_gt(1:(n_len-1),:);
        imdb.images.search_bboxs(now_index+(1:(n_len-1)),:) = bbox_gt(2:n_len,:);
        now_index = now_index+(n_len-1);
    end %%end v
end %%end alov300

% -------------------------------------------------------------------------
%                                                                     DET16
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'det16'))
    
    disp('DET16 Data:');
    det16_dataDir = opts.det16_dataDir;
    
    [img_files, ground_truth_4xy] = load_video_info_det(det16_dataDir);
    bbox_gt = get_bbox(ground_truth_4xy);
    n_len = numel(img_files);
    imdb.images.target(now_index+(1:n_len)) = img_files;
    imdb.images.search(now_index+(1:n_len)) = img_files;
    imdb.images.target_bboxs(now_index+(1:n_len),:) = bbox_gt;
    imdb.images.search_bboxs(now_index+(1:n_len),:) = bbox_gt;
    now_index = now_index+n_len;
end %%end det


dataMean(1,1,1:3) = single([123,117,104]);
imdb.images.data_mean(1,1,1:3) = dataMean;
imdb.meta.sets = {'train', 'val'} ;
end %%end function



