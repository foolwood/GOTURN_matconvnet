function imdb = vot_setup_data(varargin)
rng('default');
% opts = vl_argparse(opts, varargin) ;

opts = [];
opts.dataDir = '../data';
opts.version = 2;    % 1 vot 2 vot-lite 3 det 4 full
opts.size = [227,227];
opts = vl_argparse(opts, varargin) ;
opts.vot15_dataDir = fullfile(opts.dataDir,'VOT15');
opts.vot14_dataDir = fullfile(opts.dataDir,'VOT14');
opts.det16_dataDir = fullfile(opts.dataDir,'DET16');
opts.visualization = false;
imdb = [];
% -------------------------------------------------------------------------
%                                                                      Data
% -------------------------------------------------------------------------

switch opts.version%vot15:21395 vot14:10188
    case 1,
        set_name = {'vot15','vot14'};
        set = [ones(1,21395) 2*ones(1,10188)];
        imdb.images.set = set;
        imdb.images.target = zeros([opts.size,3,numel(set)],'single');%TODO:use name
        imdb.images.search = zeros([opts.size,3,numel(set)],'single');%TODO:use name
        imdb.images.bboxs = zeros(1,1,4,numel(set),'single');
        lite_index = 1:numel(set);
    case 2,
        set_name = {'vot15','vot14'};
        set = [ones(1,1000) 2*ones(1,1000)];%1000train_img,1000val_img
        imdb.images.set = set;
        imdb.images.target = zeros([opts.size,3,numel(set)],'single');%TODO:use name
        imdb.images.search = zeros([opts.size,3,numel(set)],'single');%TODO:use name
        imdb.images.bboxs = zeros(1,1,4,numel(set),'single');
        lite_index = zeros(1,21395+21395);
        lite_index([randperm(21395,sum(set==1)),21395+randperm(10188,sum(set==2))]) = 1:numel(set);
    case 3,
        set_name = {'det16'};
        set = [ones(1,456567) 2*ones(1,60000)];
        imdb.images.set = set;
        imdb.images.target = zeros([opts.size,3,numel(set)],'single');%TODO:use name
        imdb.images.search = zeros([opts.size,3,numel(set)],'single');%TODO:use name
        imdb.images.bboxs = zeros(1,1,4,numel(set),'single');
        lite_index = 1:numel(set);
    case 4,
        set_name = {'vot15','vot14','det16'};
        set = [ones(1,456567+21395) 2*ones(1,60000+10188)];
        imdb.images.set = set;
        imdb.images.target = zeros([opts.size,3,numel(set)],'single');%TODO:use name
        imdb.images.search = zeros([opts.size,3,numel(set)],'single');%TODO:use name
        imdb.images.bboxs = zeros(1,1,4,numel(set),'single');
        lite_index = 1:numel(set);
    otherwise,
        
end


now_index = 1;

% -------------------------------------------------------------------------
%                                                           VOT15
% -------------------------------------------------------------------------

if any(strcmpi(set_name,'vot15'))
    
    disp('VOT2015 Data(for Training):');
    dirs = dir(opts.vot15_dataDir);
    videos = {dirs.name};
    videos(strcmp('.', videos) | strcmp('..', videos)| ~[dirs.isdir]) = [];
    for  v = 1:numel(videos)
        video = videos{v};disp(['      ' video]);
        
        [img_files, ground_truth_4xy] = load_video_info_vot(opts.vot15_dataDir, video);
        bbox_gt = get_axis_aligned_BB(ground_truth_4xy);
        raw_img_bank = vl_imreadjpeg(img_files);
        
        for frame = 1:(numel(img_files)-1)
            if(lite_index(now_index))
                [image,target,bbox_gt_scaled] = ...
                    make_true_example(...
                    raw_img_bank{frame},raw_img_bank{frame+1},...
                    bbox_gt(frame,:),bbox_gt(frame+1,:),opts.size);
                
                if(opts.visualization)
                    subplot(1,2,1);imshow(target);subplot(1,2,2);imshow(image);
                    disp(bbox_gt_scaled);
                    drawnow;
                end
                
                imdb.images.target(:,:,:,lite_index(now_index)) = single(target);
                imdb.images.search(:,:,:,lite_index(now_index)) = single(image);
                imdb.images.bboxs(1,1,1:4,lite_index(now_index)) =...
                    single(bbox_gt_scaled) ;
                
                
            end
            now_index = now_index+1;
        end %%end frame
    end %%end v
end %%end vot15


% -------------------------------------------------------------------------
%                                                           VOT14
% -------------------------------------------------------------------------

if any(strcmpi(set_name,'vot14'))
    
    disp('VOT2014 Data(for Validation):');
    dirs = dir(opts.vot14_dataDir);
    videos = {dirs.name};
    videos(strcmp('.', videos) | strcmp('..', videos)| ~[dirs.isdir]) = [];
    for  v = 1:numel(videos)
        video = videos{v};disp(['      ' video]);
        
        [img_files, ground_truth_4xy] = load_video_info_vot(opts.vot14_dataDir, video);
        bbox_gt = get_axis_aligned_BB(ground_truth_4xy);
        raw_img_bank = vl_imreadjpeg(img_files);
        
        for frame = 1:(numel(img_files)-1)
            if(lite_index(now_index))
                
                [image,target,bbox_gt_scaled] = ...
                    make_true_example(...
                    raw_img_bank{frame},raw_img_bank{frame+1},...
                    bbox_gt(frame,:),bbox_gt(frame+1,:),opts.size);
                
                if(opts.visualization)
                    subplot(1,2,1);imshow(target);subplot(1,2,2);imshow(image);
                    disp(bbox_gt_scaled);
                    drawnow;
                end
                
                imdb.images.target(:,:,:,lite_index(now_index)) = single(target);
                imdb.images.search(:,:,:,lite_index(now_index)) = single(image);
                imdb.images.bboxs(1,1,1:4,lite_index(now_index)) =...
                    single(bbox_gt_scaled) ;
                
            end
            now_index = now_index+1;
        end %%end frame
    end %%end v
end %%end vot14

% dataMean = mean(imdb.images.target(:,:,:,set == 1), 4);
dataMean(1,1,1:3) = single([123,117,104]);
imdb.images.target = bsxfun(@minus, imdb.images.target, dataMean) ;
imdb.images.search = bsxfun(@minus, imdb.images.search, dataMean) ;

imdb.images.data_mean(1,1,1:3) = dataMean;
imdb.images.size = opts.size;
imdb.meta.sets = {'train', 'val'} ;
end %%end function



% -------------------------------------------------------------------------------------------------
function [bb] = get_axis_aligned_BB(region)
%GETAXISALIGNEDBB computes axis-aligned bbox with same area as the rotated one (REGION)
% -------------------------------------------------------------------------------------------------

% cx = mean(region(:,1:2:end),2);
% cy = mean(region(:,2:2:end),2);
% x1 = min(region(:,1:2:end),[],2);
% x2 = max(region(:,1:2:end),[],2);
% y1 = min(region(:,2:2:end),[],2);
% y2 = max(region(:,2:2:end),[],2);
% x1y1x2y2 = region(:,1:2) - region(:,3:4);
% x2y2x3y3 = region(:,3:4) - region(:,5:6);
% A1 = sqrt(sum(x1y1x2y2.*x1y1x2y2,2)).* sqrt(sum(x2y2x3y3.*x2y2x3y3,2));
% A2 = (x2 - x1) .* (y2 - y1);
% s = sqrt(A1./A2);
% w = s .* (x2 - x1) + 1;
% h = s .* (y2 - y1) + 1;
% bb = [cx-w/2,cy-h/2,cx+w/2,cy+h/2]-1;
bb = [min(region(:,1:2:end),[],2),...
    min(region(:,2:2:end),[],2),...
    max(region(:,1:2:end),[],2),...
    max(region(:,2:2:end),[],2)]-1;

% rect = [cx-w/2,cy-h/2,w,h];
% rect = [min(region(:,1:2:end),[],2),...
%         min(region(:,2:2:end),[],2),...
%         max(region(:,1:2:end),[],2)-...
%         min(region(:,1:2:end),[],2),...
%         max(region(:,2:2:end),[],2)-...
%         min(region(:,2:2:end),[],2)];

end
