function imdb = setup_data_ram(varargin)
rng('default');
addpath('../utils');
opts = [];
opts.dataDir = '../data';
opts.version = 1;
opts = vl_argparse(opts, varargin) ;

opts.vot16_dataDir = fullfile(opts.dataDir,'VOT16');

opts.visualization = false;
imdb = [];

% -------------------------------------------------------------------------
%           vot16:21395 vot15:21395 vot14:10188 vot13:5665
%           cvpr2013:29435 tb100:58935 tb50:26922
%           nus_pro:26090 tc128:55217 alov300:16023
% -------------------------------------------------------------------------

switch opts.version
    case 1,
        bbox_mode = 'minmax';%
        set_name = {'vot16'};
        set = ones(1,21395);
        set(randperm(21395,1000)) = 2;
    otherwise,
        
end


if strcmp(bbox_mode,'axis_aligned'),
    get_bbox = @get_axis_aligned_BB;
else
    get_bbox = @get_minmax_BB;
end

imdb.images.set = set;
imdb.images.target = cell([numel(set),1]);
imdb.images.search = cell([numel(set),1]);
imdb.images.target_bboxs = zeros(1,1,4,numel(set),'single');
imdb.images.search_bboxs = zeros(1,1,4,numel(set),'single');

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
        bbox_gt = reshape(bbox_gt',1,1,4,[]);
        n_len = numel(img_files);
        img_bank = vl_imreadjpeg(img_files);
        imdb.images.target(now_index+(1:(n_len-1))) = img_bank(1:(n_len-1));
        imdb.images.search(now_index+(1:(n_len-1))) = img_bank(2:n_len);
        imdb.images.target_bboxs(1,1,1:4,now_index+(1:(n_len-1))) = single(bbox_gt(:,:,:,1:(n_len-1)));
        imdb.images.search_bboxs(1,1,1:4,now_index+(1:(n_len-1))) = single(bbox_gt(:,:,:,2:n_len));
        now_index = now_index+(n_len-1);
    end %%end v
end %%end vot16



dataMean(1,1,1:3) = single([123,117,104]);
imdb.images.data_mean(1,1,1:3) = dataMean;
imdb.meta.sets = {'train', 'val'} ;
end %%end function



