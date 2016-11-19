function imdb = setup_data(varargin)
rng('default');
addpath('../utils');
addpath('../data/ILSVRC/devkit/evaluation')
opts = [];
opts.dataDir = '../data';
opts.version = 10;
opts = vl_argparse(opts, varargin) ;
opts.expDir = ['../data/crop' num2str(opts.version)];

opts.vot16_dataDir = fullfile(opts.dataDir,'VOT16');
opts.vot15_dataDir = fullfile(opts.dataDir,'VOT15');
opts.vot14_dataDir = fullfile(opts.dataDir,'VOT14');
opts.vot13_dataDir = fullfile(opts.dataDir,'VOT13');

opts.otb_dataDir = fullfile(opts.dataDir,'OTB');

opts.nus_pro_dataDir = fullfile(opts.dataDir,'NUS_PRO');

opts.tc128_dataDir = fullfile(opts.dataDir,'TC128');

opts.alov300_dataDir = fullfile(opts.dataDir,'ALOV300');

opts.ilsvrc_dataDir = fullfile(opts.dataDir,'ILSVRC');

opts.visualization = false;
imdb = [];

% -------------------------------------------------------------------------
%           vot16:21395 vot15:21395 vot14:10188 vot13:5665
%           cvpr2013:29435 tb100:58935 tb50:26922
%           nus_pro:26090 tc128:55217 alov300:89351
% -------------------------------------------------------------------------

switch opts.version
    case 1,
        nsample = 1;
        bbox_mode = 'minmax';%
        set_name = {'vot15','vot14'};
        set = [ones(1,21395*nsample) 2*ones(1,10188*1)];
    case 2,
        nsample = 1;
        bbox_mode = 'axis_aligned';%
        set_name = {'vot15','vot14'};
        set = [ones(1,21395*nsample) 2*ones(1,10188*1)];
    case 3,
        nsample = 1;
        bbox_mode = 'axis_aligned';%
        set_name = {'vot15','vot14','nus_pro'};
        set = [ones(1,21395*nsample) 2*ones(1,10188*1) ones(1,26090*nsample)];
    case 4,
        nsample = 10;
        bbox_mode = 'axis_aligned';%
        set_name = {'vot15','vot14','nus_pro'};
        set = [ones(1,21395*nsample) 2*ones(1,10188*1) ones(1,26090*nsample)];
    case 5,
        nsample = 20;
        bbox_mode = 'axis_aligned';%
        set_name = {'vot15','vot14','nus_pro'};
        set = [ones(1,21395*nsample) 2*ones(1,10188*1) ones(1,26090*nsample)];
    case 6,
        nsample = 30;
        bbox_mode = 'axis_aligned';%
        set_name = {'vot15','vot14','nus_pro'};
        set = [ones(1,21395*nsample) 2*ones(1,10188*1) ones(1,26090*nsample)];
    case 7,
        nsample = 40;
        bbox_mode = 'axis_aligned';%
        set_name = {'vot15','vot14','nus_pro'};
        set = [ones(1,21395*nsample) 2*ones(1,10188*1) ones(1,26090*nsample)];
    case 8,
        nsample = 50;
        bbox_mode = 'axis_aligned';%
        set_name = {'vot15','vot14','nus_pro'};
        set = [ones(1,21395*nsample) 2*ones(1,10188*1) ones(1,26090*nsample)];
    case 9,
        nsample = 1;
        bbox_mode = 'axis_aligned';%
        set_name = {'vot16_no_cvpr2013','nus_pro',...
            'tc128_no_cvpr2013','cvpr2013'};
        set = [ones(1,17695*nsample),ones(1,26090*1),ones(1,30507*nsample)...
            2*ones(1,29435*nsample)];
    case 10,
        nsample = 10;
        bbox_mode = 'axis_aligned';%
        set_name = {...%'vot16','vot15','vot14','vot13',
            'cvpr2013',...
            'tb100','tb50','nus_pro','tc128','alov300','ilsvrc'};
        set = [ones(1,100000*nsample)];
    otherwise,
        
end


if strcmp(bbox_mode,'axis_aligned'),
    get_bbox = @get_axis_aligned_BB;
else
    get_bbox = @get_minmax_BB;
end

imdb.images.set = set;
imdb.images.target = cell([numel(set),1]);
imdb.images.image = cell([numel(set),1]);
imdb.images.bboxs = zeros(1,1,4,numel(set),'single');

now_index = 0;
expDir = opts.expDir;
expDir(expDir == '\') = '/';%for sprintf
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
        im_bank = vl_imreadjpeg(img_files);
        video_expDir = [expDir '/vot16/' video];
        if ~exist(video_expDir,'dir'),mkdir(video_expDir) ;end;
        for frame = 1:(numel(im_bank)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            make_all_examples(im_bank{frame},im_bank{frame+1},...
                bbox_gt(frame,:),bbox_gt(frame+1,:),nsample,video_frame_expDir);
            
            load([sprintf(video_frame_expDir,0,0),'.mat']);
            imdb.images.bboxs(1,1,1:4,now_index+(1:nsample)) = bbox_gt_scaled;
            for k = 1:nsample
                imdb.images.target(now_index+k) = {[sprintf(video_frame_expDir,0,k),'.jpg']};
                imdb.images.image(now_index+k) = {[sprintf(video_frame_expDir,1,k),'.jpg']};
            end
            now_index = now_index+nsample;
        end %%end frame
    end %%end v
end %%end vot16

% -------------------------------------------------------------------------
%                                                                     VOT15
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'vot15'))
    
    disp('VOT2015 Data:');
    vot15_dataDir = opts.vot15_dataDir;
    videos = importdata(fullfile(vot15_dataDir,'list.txt'));
    
    for v = 1:numel(videos)
        video = videos{v};fprintf('%3d :%20s\n',v,video);
        [img_files, ground_truth_4xy] = load_video_info_vot(vot15_dataDir, video);
        bbox_gt = get_bbox(ground_truth_4xy);
        im_bank = vl_imreadjpeg(img_files);
        video_expDir = [expDir '/vot15/' video];
        if ~exist(video_expDir,'dir'),mkdir(video_expDir) ;end;
        for frame = 1:(numel(im_bank)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            make_all_examples(im_bank{frame},im_bank{frame+1},...
                bbox_gt(frame,:),bbox_gt(frame+1,:),nsample,video_frame_expDir);
            
            load([sprintf(video_frame_expDir,0,0),'.mat']);
            imdb.images.bboxs(1,1,1:4,now_index+(1:nsample)) = bbox_gt_scaled;
            for k = 1:nsample
                imdb.images.target(now_index+k) = {[sprintf(video_frame_expDir,0,k),'.jpg']};
                imdb.images.image(now_index+k) = {[sprintf(video_frame_expDir,1,k),'.jpg']};
            end
            now_index = now_index+nsample;
        end %%end frame
    end %%end v
end %%end vot15


% -------------------------------------------------------------------------
%                                                                     VOT14
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'vot14'))
    
    disp('VOT2014 Data:');
    vot14_dataDir = opts.vot14_dataDir;
    videos = importdata(fullfile(vot14_dataDir,'list.txt'));
    
    for v = 1:numel(videos)
        video = videos{v};fprintf('%3d :%20s\n',v,video);
        [img_files, ground_truth_4xy] = load_video_info_vot(vot14_dataDir, video);
        bbox_gt = get_bbox(ground_truth_4xy);
        im_bank = vl_imreadjpeg(img_files);
        video_expDir = [expDir '/vot14/' video];
        if ~exist(video_expDir,'dir'),mkdir(video_expDir) ;end;
        for frame = 1:(numel(im_bank)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            make_all_examples(im_bank{frame},im_bank{frame+1},...
                bbox_gt(frame,:),bbox_gt(frame+1,:),nsample,video_frame_expDir);
            
            load([sprintf(video_frame_expDir,0,0),'.mat']);
            imdb.images.bboxs(1,1,1:4,now_index+(1:nsample)) = bbox_gt_scaled;
            for k = 1:nsample
                imdb.images.target(now_index+k) = {[sprintf(video_frame_expDir,0,k),'.jpg']};
                imdb.images.image(now_index+k) = {[sprintf(video_frame_expDir,1,k),'.jpg']};
            end
            now_index = now_index+nsample;
        end %%end frame
    end %%end v
end %%end vot14

% -------------------------------------------------------------------------
%                                                                     VOT13
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'vot13'))
    
    disp('VOT2013 Data:');
    vot13_dataDir = opts.vot13_dataDir;
    videos = importdata(fullfile(vot13_dataDir,'list.txt'));
    
    for v = 1:numel(videos)
        video = videos{v};fprintf('%3d :%20s\n',v,video);
        [img_files, ground_truth_4xy] = load_video_info_vot(vot13_dataDir, video);
        bbox_gt = get_bbox(ground_truth_4xy);
        im_bank = vl_imreadjpeg(img_files);
        video_expDir = [expDir '/vot13/' video];
        if ~exist(video_expDir,'dir'),mkdir(video_expDir) ;end;
        for frame = 1:(numel(im_bank)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            make_all_examples(im_bank{frame},im_bank{frame+1},...
                bbox_gt(frame,:),bbox_gt(frame+1,:),nsample,video_frame_expDir);
            
            load([sprintf(video_frame_expDir,0,0),'.mat']);
            imdb.images.bboxs(1,1,1:4,now_index+(1:nsample)) = bbox_gt_scaled;
            for k = 1:nsample
                imdb.images.target(now_index+k) = {[sprintf(video_frame_expDir,0,k),'.jpg']};
                imdb.images.image(now_index+k) = {[sprintf(video_frame_expDir,1,k),'.jpg']};
            end
            now_index = now_index+nsample;
        end %%end frame
    end %%end v
end %%end vot13

% -------------------------------------------------------------------------
%                                                                  CVPR2013
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'cvpr2013'))
    
    disp('CVPR2013 Data:');
    cvpr2013_dataDir = opts.otb_dataDir;
    CVPR2013 = readtable(fullfile(cvpr2013_dataDir,'cvpr13.txt'),'ReadVariableNames',false);
    videos = CVPR2013.Var1;
    
    for  v = 1:numel(videos)
        video = videos{v};fprintf('%3d :%20s\n',v,video);
        [img_files, ground_truth_4xy] = load_video_info_otb(cvpr2013_dataDir, video);
        bbox_gt = get_bbox(ground_truth_4xy);
        im_bank = vl_imreadjpeg(img_files);
        video_expDir = [expDir '/cvpr2013/' video];
        if ~exist(video_expDir,'dir'),mkdir(video_expDir) ;end;
        for frame = 1:(numel(im_bank)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            make_all_examples(im_bank{frame},im_bank{frame+1},...
                bbox_gt(frame,:),bbox_gt(frame+1,:),nsample,video_frame_expDir);
            
            load([sprintf(video_frame_expDir,0,0),'.mat']);
            imdb.images.bboxs(1,1,1:4,now_index+(1:nsample)) = bbox_gt_scaled;
            for k = 1:nsample
                imdb.images.target(now_index+k) = {[sprintf(video_frame_expDir,0,k),'.jpg']};
                imdb.images.image(now_index+k) = {[sprintf(video_frame_expDir,1,k),'.jpg']};
            end
            now_index = now_index+nsample;
        end %%end frame
    end %%end v
end %%end cvpr2013

% -------------------------------------------------------------------------
%                                                                     TB100
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'tb100'))
    
    disp('TB100 Data:');
    tb100_dataDir = opts.otb_dataDir;
    TB100 = readtable(fullfile(tb100_dataDir,'tb_100.txt'),'ReadVariableNames',false);
    videos = TB100.Var1;
    
    for  v = 1:numel(videos)
        video = videos{v};fprintf('%3d :%20s\n',v,video);
        [img_files, ground_truth_4xy] = load_video_info_otb(tb100_dataDir, video);
        bbox_gt = get_bbox(ground_truth_4xy);
        im_bank = vl_imreadjpeg(img_files);
        video_expDir = [expDir '/tb100/' video];
        if ~exist(video_expDir,'dir'),mkdir(video_expDir) ;end;
        for frame = 1:(numel(im_bank)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            make_all_examples(im_bank{frame},im_bank{frame+1},...
                bbox_gt(frame,:),bbox_gt(frame+1,:),nsample,video_frame_expDir);
            
            load([sprintf(video_frame_expDir,0,0),'.mat']);
            imdb.images.bboxs(1,1,1:4,now_index+(1:nsample)) = bbox_gt_scaled;
            for k = 1:nsample
                imdb.images.target(now_index+k) = {[sprintf(video_frame_expDir,0,k),'.jpg']};
                imdb.images.image(now_index+k) = {[sprintf(video_frame_expDir,1,k),'.jpg']};
            end
            now_index = now_index+nsample;
        end %%end frame
    end %%end v
end %%end tb100

% -------------------------------------------------------------------------
%                                                                      TB50
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'tb50'))
    
    disp('TB50 Data:');
    tb50_dataDir = opts.otb_dataDir;
    TB50 = readtable(fullfile(tb50_dataDir,'tb_50.txt'),'ReadVariableNames',false);
    videos = TB50.Var1;
    
    for  v = 1:numel(videos)
        video = videos{v};fprintf('%3d :%20s\n',v,video);
        [img_files, ground_truth_4xy] = load_video_info_otb(tb50_dataDir, video);
        bbox_gt = get_bbox(ground_truth_4xy);
        im_bank = vl_imreadjpeg(img_files);
        video_expDir = [expDir '/tb50/' video];
        if ~exist(video_expDir,'dir'),mkdir(video_expDir) ;end;
        for frame = 1:(numel(im_bank)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            make_all_examples(im_bank{frame},im_bank{frame+1},...
                bbox_gt(frame,:),bbox_gt(frame+1,:),nsample,video_frame_expDir);
            
            load([sprintf(video_frame_expDir,0,0),'.mat']);
            imdb.images.bboxs(1,1,1:4,now_index+(1:nsample)) = bbox_gt_scaled;
            for k = 1:nsample
                imdb.images.target(now_index+k) = {[sprintf(video_frame_expDir,0,k),'.jpg']};
                imdb.images.image(now_index+k) = {[sprintf(video_frame_expDir,1,k),'.jpg']};
            end
            now_index = now_index+nsample;
        end %%end frame
    end %%end v
end %%end tb50

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
        im_bank = vl_imreadjpeg(img_files);
        video_expDir = [expDir '/nus_pro/' video];
        if ~exist(video_expDir,'dir'),mkdir(video_expDir) ;end;
        for frame = 1:(numel(im_bank)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            make_all_examples(im_bank{frame},im_bank{frame+1},...
                bbox_gt(frame,:),bbox_gt(frame+1,:),nsample,video_frame_expDir);
            
            load([sprintf(video_frame_expDir,0,0),'.mat']);
            imdb.images.bboxs(1,1,1:4,now_index+(1:nsample)) = bbox_gt_scaled;
            for k = 1:nsample
                imdb.images.target(now_index+k) = {[sprintf(video_frame_expDir,0,k),'.jpg']};
                imdb.images.image(now_index+k) = {[sprintf(video_frame_expDir,1,k),'.jpg']};
            end
            now_index = now_index+nsample;
        end %%end frame
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
        im_bank = vl_imreadjpeg(img_files);
        video_expDir = [expDir '/tc128/' video];
        if ~exist(video_expDir,'dir'),mkdir(video_expDir) ;end;
        for frame = 1:(numel(im_bank)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            make_all_examples(im_bank{frame},im_bank{frame+1},...
                bbox_gt(frame,:),bbox_gt(frame+1,:),nsample,video_frame_expDir);
            
            load([sprintf(video_frame_expDir,0,0),'.mat']);
            imdb.images.bboxs(1,1,1:4,now_index+(1:nsample)) = bbox_gt_scaled;
            for k = 1:nsample
                imdb.images.target(now_index+k) = {[sprintf(video_frame_expDir,0,k),'.jpg']};
                imdb.images.image(now_index+k) = {[sprintf(video_frame_expDir,1,k),'.jpg']};
            end
            now_index = now_index+nsample;
        end %%end frame
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
        im_bank = vl_imreadjpeg(img_files);
        video_expDir = [expDir '/alov300/' video];
        if ~exist(video_expDir,'dir'),mkdir(video_expDir) ;end;
        for frame = 1:(numel(im_bank)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            make_all_examples(im_bank{frame},im_bank{frame+1},...
                bbox_gt(frame,:),bbox_gt(frame+1,:),nsample,video_frame_expDir);
            
            load([sprintf(video_frame_expDir,0,0),'.mat']);
            imdb.images.bboxs(1,1,1:4,now_index+(1:nsample)) = bbox_gt_scaled;
            for k = 1:nsample
                imdb.images.target(now_index+k) = {[sprintf(video_frame_expDir,0,k),'.jpg']};
                imdb.images.image(now_index+k) = {[sprintf(video_frame_expDir,1,k),'.jpg']};
            end
            now_index = now_index+nsample;
        end %%end frame
    end %%end v
end %%end alov300

% -------------------------------------------------------------------------
%                                                                    ILSVRC
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'ilsvrc'))
    
    disp('ILSVRC Data(VID):');
    ilsvrc_dataDir = opts.ilsvrc_dataDir;
    ILSVRC_train_temp1 = dir(fullfile(ilsvrc_dataDir,...
        'Data','VID','train','ILSVRC2015_VID_train_*'));
    ILSVRC_train_temp1 = sort({ILSVRC_train_temp1.name});
    ILSVRC_train = cell(0);
    for k = 1:numel(ILSVRC_train_temp1)
        ILSVRC_train_temp2 = dir(fullfile(ilsvrc_dataDir,...
            'Data','VID','train',ILSVRC_train_temp1{k},'ILSVRC2015_train_*'));
        ILSVRC_train_temp2 = sort({ILSVRC_train_temp2.name});
        ILSVRC_train(end+(1:numel(ILSVRC_train_temp2))) = fullfile(ILSVRC_train_temp1{k},ILSVRC_train_temp2);
    end
    ILSVRC_val = dir('../ILSVRC/Data/VID/val/ILSVRC2015_val_*');
    ILSVRC_val = sort({ILSVRC_val.name});
    ILSVRC = [ILSVRC_train,ILSVRC_val];
    videos = ILSVRC;
    
    for  v = 1:numel(videos)
        video = videos{v};video(video == '\') = '/';fprintf('%3d :%20s\n',v,video);
        [img_files, ~, vid_info] = load_video_info_vid(ilsvrc_dataDir, video);
        im_bank = vl_imreadjpeg(img_files);
        video_expDir = [expDir '/ilsvrc/' video];
        
        for frame = 1:(numel(im_bank)-1)
            for i = 1:numel(vid_info{frame}.trackid)
                for j = 1:numel(vid_info{frame+1}.trackid)
                    if (vid_info{frame}.trackid(i) == vid_info{frame+1}.trackid(j))
                        if ~exist(video_expDir,'dir'),mkdir(video_expDir) ;end;
                        video_frame_expDir = [video_expDir '/' num2str(frame) '-' num2str(i) '-%d-%d' ];
                        make_all_examples(im_bank{frame},im_bank{frame+1},...
                            vid_info{frame}.bbox(i,:),vid_info{frame+1}.bbox(j,:),nsample,video_frame_expDir);
                        
                        load([sprintf(video_frame_expDir,0,0),'.mat']);
                        imdb.images.bboxs(1,1,1:4,now_index+(1:nsample)) = bbox_gt_scaled;
                        for k = 1:nsample
                            imdb.images.target(now_index+k) = {[sprintf(video_frame_expDir,0,k),'.jpg']};
                            imdb.images.image(now_index+k) = {[sprintf(video_frame_expDir,1,k),'.jpg']};
                        end
                        now_index = now_index+nsample;
                    end
                end
            end
        end %%end frame
    end %%end v
end %%end ilsvrc2016

% -------------------------------------------------------------------------
%                                                         vot16_no_cvpr2013
% -------------------------------------------------------------------------
if any(strcmpi(set_name,'vot16_no_cvpr2013'))
    
    disp('VOT2016 Data(no cvpr2013):');
    vot16_dataDir = opts.vot16_dataDir;
    videos = {'bag','ball1','ball2','birds1','birds2','blanket','bmx',...
        'bolt2','book','butterfly','car1','crossing','dinosaur',...
        'fernando','fish1','fish2','fish3','fish4','girl','glove',...
        'godfather','graduate','gymnastics1','gymnastics2',...
        'gymnastics3','gymnastics4','hand','handball1','handball2',...
        'helicopter','iceskater1','iceskater2','leaves','marching',...
        'motocross2','nature','octopus','pedestrian2','rabbit','racing',...
        'road','sheep','singer3','soccer2','soldier','sphere','traffic',...
        'tunnel','wiper'};
    
    for  v = 1:numel(videos)
        video = videos{v};fprintf('%3d :%20s\n',v,video);
        [img_files, ground_truth_4xy] = load_video_info_vot(vot16_dataDir, video);
        bbox_gt = get_bbox(ground_truth_4xy);
        im_bank = vl_imreadjpeg(img_files);
        video_expDir = [expDir '/vot16_no_cvpr2013/' video];
        if ~exist(video_expDir,'dir'),mkdir(video_expDir) ;end;
        for frame = 1:(numel(im_bank)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            make_all_examples(im_bank{frame},im_bank{frame+1},...
                bbox_gt(frame,:),bbox_gt(frame+1,:),nsample,video_frame_expDir);
            
            load([sprintf(video_frame_expDir,0,0),'.mat']);
            imdb.images.bboxs(1,1,1:4,now_index+(1:nsample)) = bbox_gt_scaled;
            for k = 1:nsample
                imdb.images.target(now_index+k) = {[sprintf(video_frame_expDir,0,k),'.jpg']};
                imdb.images.image(now_index+k) = {[sprintf(video_frame_expDir,1,k),'.jpg']};
            end
            now_index = now_index+nsample;
        end %%end frame
    end %%end v
end %%end vot16_no_cvpr2013

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
        im_bank = vl_imreadjpeg(img_files);
        video_expDir = [expDir '/tc128_no_cvpr2013/' video];
        if ~exist(video_expDir,'dir'),mkdir(video_expDir) ;end;
        for frame = 1:(numel(im_bank)-1)
            video_frame_expDir = [video_expDir '/' num2str(frame) '-%d-%d' ];
            make_all_examples(im_bank{frame},im_bank{frame+1},...
                bbox_gt(frame,:),bbox_gt(frame+1,:),nsample,video_frame_expDir);
            
            load([sprintf(video_frame_expDir,0,0),'.mat']);
            imdb.images.bboxs(1,1,1:4,now_index+(1:nsample)) = bbox_gt_scaled;
            for k = 1:nsample
                imdb.images.target(now_index+k) = {[sprintf(video_frame_expDir,0,k),'.jpg']};
                imdb.images.image(now_index+k) = {[sprintf(video_frame_expDir,1,k),'.jpg']};
            end
            now_index = now_index+nsample;
        end %%end frame
    end %%end v
end %%end tc128_no_cvpr2013

dataMean(1,1,1:3) = single([123,117,104]);
imdb.images.data_mean(1,1,1:3) = dataMean;
imdb.meta.sets = {'train', 'val'} ;
end %%end function



