function imdb = vot_setup_data(varargin)

% opts = vl_argparse(opts, varargin) ;

opts = [];
opts.size = [227,227];
kScaleFactor = 10;
opts.dataDir = '../data';
opts = vl_argparse(opts, varargin) ;
opts.train_dataDir = fullfile(opts.dataDir,'VOT15');
opts.val_dataDir = fullfile(opts.dataDir,'VOT14');

opts.show_visualization = false;
opts.make_imdb = true;%make sure it will take a looooong time & memory
opts.lite = ismac();

imdb = [];
% -------------------------------------------------------------------------
%                                                             Training Data
% -------------------------------------------------------------------------


if(opts.make_imdb)%vot15:21395 vot14:10188
    if(~opts.lite)
        set = [ones(1,21395) 2*ones(1,10188)];
        imdb.images.set = set;
        imdb.images.target = zeros(opts.size(1),opts.size(2),3,...
            numel(set),'single');
        imdb.images.search = zeros(opts.size(1),opts.size(2),3,...
            numel(set),'single');
        imdb.images.bboxs = zeros(1,1,4,numel(set),'single');
        lite_index = zeros(1,21395+10188);
        lite_index([randperm(21395,sum(set==1)),21395+randperm(10188,sum(set==2))]) = 1:numel(set);
    else
        set = [ones(1,1000) 2*ones(1,1000)];
        imdb.images.set = set;
        imdb.images.target = zeros(opts.size(1),opts.size(2),3,...
            numel(set),'single');
        imdb.images.search = zeros(opts.size(1),opts.size(2),3,...
            numel(set),'single');
        imdb.images.bboxs = zeros(1,1,4,numel(set),'single');
        lite_index = zeros(1,21395+10188);
        lite_index([randperm(21395,sum(set==1)),21395+randperm(10188,sum(set==2))]) = 1:numel(set);
    end
end





now_index = 1;
bboxs113 = zeros(1,1,4,'single');

disp('Training Data:');
dirs = dir(opts.train_dataDir);
videos = {dirs.name};
videos(strcmp('.', videos) | strcmp('..', videos)| ~[dirs.isdir]) = [];
for  v = 1:numel(videos)
    video = videos{v};disp(['      ' video]);
    
    [img_files, ground_truth_4xy] = load_video_info_vot(opts.train_dataDir, video);
    ground_truth_rect = get_axis_aligned_BB(ground_truth_4xy);
    raw_img_bank = vl_imreadjpeg(img_files);
    im_cols = size(raw_img_bank{1},2);
    im_rows = size(raw_img_bank{1},1);
    
    rect_target = ground_truth_rect(1:end-1,:);
    rect_result = ground_truth_rect(2:end,:);
    
    ltrb_pad2 = [rect_target(:,1)-rect_target(:,3)/2,...
        rect_target(:,2)-rect_target(:,4)/2,...
        rect_target(:,1)+rect_target(:,3)*3/2,...
        rect_target(:,2)+rect_target(:,4)*3/2];
    ltrb_result = [rect_result(:,1:2),rect_result(:,1:2)+rect_result(:,3:4)];
    xs_start = round(ltrb_pad2(:,1));
    xs_stop = round(ltrb_pad2(:,3));
    
    ys_start = round(ltrb_pad2(:,2));
    ys_stop = round(ltrb_pad2(:,4));
    
    ltrb_result_in_pad2 = ltrb_result-[xs_start,ys_start,xs_start,ys_start];
    
    w = xs_stop-xs_start+1;h = ys_stop-ys_start+1;
    
    ltrb_result_in_im_search = (ltrb_result_in_pad2.*opts.size(1)./[w,h,w,h]);
    
    bboxs = (ltrb_result_in_pad2.*kScaleFactor./[w,h,w,h]);
    
    for frame = 1:(numel(img_files)-1)
        if(lite_index(now_index))
            raw_im_target = raw_img_bank{frame};
            raw_im_search = raw_img_bank{frame+1};
            
            xs = xs_start(frame):xs_stop(frame);
            ys = ys_start(frame):ys_stop(frame);
            
            xs(xs < 1) = 1;
            ys(ys < 1) = 1;
            xs(xs > im_cols) = im_cols;
            ys(ys > im_rows) = im_rows;
            
            crop_im_target = raw_im_target(ys, xs, :);
            crop_im_search = raw_im_search(ys, xs, :);
            im_target = imresize(crop_im_target,opts.size);
            im_search = imresize(crop_im_search,opts.size);
            
            if(opts.show_visualization)
                subplot(2,2,1);imshow(uint8(im_target));
                subplot(2,2,2);imshow(uint8(im_search));
                hold on ;
                rectangle('Position',[ltrb_result_in_im_search(frame,1:2),...
                    ltrb_result_in_im_search(frame,3:4)-ltrb_result_in_im_search(frame,1:2)]);
                hold off ;
                
                subplot(2,2,3);imshow(uint8(crop_im_target));
                subplot(2,2,4);imshow(uint8(crop_im_search));
                hold on ;
                rectangle('Position',[ltrb_result_in_pad2(frame,1:2),...
                    ltrb_result_in_pad2(frame,3:4)-ltrb_result_in_pad2(frame,1:2)]);
                hold off ;
                drawnow;
            end
            
            if(opts.make_imdb)
                imdb.images.target(:,:,:,lite_index(now_index)) = single(im_target);
                imdb.images.search(:,:,:,lite_index(now_index)) = single(im_search);
                bboxs113(1:4) = bboxs(frame,1:4);
                imdb.images.bboxs(:,:,:,lite_index(now_index)) = single(bboxs113) ;
            end
        end
        now_index = now_index+1;
    end %%end frame
end %%end v


% -------------------------------------------------------------------------
%                                                           Validation Data
% -------------------------------------------------------------------------

disp('Validation Data:');
dirs = dir(opts.val_dataDir);
videos = {dirs.name};
videos(strcmp('.', videos) | strcmp('..', videos)| ~[dirs.isdir]) = [];
for  v = 1:numel(videos)
    video = videos{v};disp(['      ' video]);
    
    [img_files, ground_truth_4xy] = load_video_info_vot(opts.val_dataDir, video);
    ground_truth_rect = get_axis_aligned_BB(ground_truth_4xy);
    raw_img_bank = vl_imreadjpeg(img_files);
    im_cols = size(raw_img_bank{1},2);
    im_rows = size(raw_img_bank{1},1);
    
    rect_target = ground_truth_rect(1:end-1,:);
    rect_result = ground_truth_rect(2:end,:);
    
    ltrb_pad2 = [rect_target(:,1)-rect_target(:,3)/2,...
        rect_target(:,2)-rect_target(:,4)/2,...
        rect_target(:,1)+rect_target(:,3)*3/2,...
        rect_target(:,2)+rect_target(:,4)*3/2];
    ltrb_result = [rect_result(:,1:2),rect_result(:,1:2)+rect_result(:,3:4)];
    xs_start = round(ltrb_pad2(:,1));
    xs_stop = round(ltrb_pad2(:,3));
    
    ys_start = round(ltrb_pad2(:,2));
    ys_stop = round(ltrb_pad2(:,4));
    
    ltrb_result_in_pad2 = ltrb_result-[xs_start,ys_start,xs_start,ys_start];
    
    w = xs_stop-xs_start+1;h = ys_stop-ys_start+1;
    
    ltrb_result_in_im_search = (ltrb_result_in_pad2.*opts.size(1)./[w,h,w,h]);
    
    bboxs = (ltrb_result_in_pad2.*kScaleFactor./[w,h,w,h]);
    
    for frame = 1:(numel(img_files)-1)
        if(lite_index(now_index))
            raw_im_target = raw_img_bank{frame};
            raw_im_search = raw_img_bank{frame+1};
            
            xs = xs_start(frame):xs_stop(frame);
            ys = ys_start(frame):ys_stop(frame);
            
            xs(xs < 1) = 1;
            ys(ys < 1) = 1;
            xs(xs > im_cols) = im_cols;
            ys(ys > im_rows) = im_rows;
            
            crop_im_target = raw_im_target(ys, xs, :);
            crop_im_search = raw_im_search(ys, xs, :);
            im_target = imresize(crop_im_target,opts.size);
            im_search = imresize(crop_im_search,opts.size);
            
            if(opts.show_visualization)
                subplot(2,2,1);imshow(uint8(im_target));
                subplot(2,2,2);imshow(uint8(im_search));
                hold on ;
                rectangle('Position',[ltrb_result_in_im_search(frame,1:2),...
                    ltrb_result_in_im_search(frame,3:4)-ltrb_result_in_im_search(frame,1:2)]);
                hold off ;
                
                subplot(2,2,3);imshow(uint8(crop_im_target));
                subplot(2,2,4);imshow(uint8(crop_im_search));
                hold on ;
                rectangle('Position',[ltrb_result_in_pad2(frame,1:2),...
                    ltrb_result_in_pad2(frame,3:4)-ltrb_result_in_pad2(frame,1:2)]);
                hold off ;
                drawnow;
            end
            
            if(opts.make_imdb)
                imdb.images.target(:,:,:,lite_index(now_index)) = single(im_target);
                imdb.images.search(:,:,:,lite_index(now_index)) = single(im_search);
                bboxs113(1:4) = bboxs(frame,1:4);
                imdb.images.bboxs(:,:,:,lite_index(now_index)) = single(bboxs113) ;
            end
        end
        now_index = now_index+1;
    end %%end frame
end %%end v

if(opts.make_imdb ||opts.lite)
    dataMean = mean(cat(4,imdb.images.target(:,:,:,set == 1),...
        imdb.images.search(:,:,:,set == 1)),4);
    imdb.images.data_mean = dataMean;
    imdb.images.target = bsxfun(@minus, imdb.images.target, dataMean) ;
    imdb.images.search = bsxfun(@minus, imdb.images.search, dataMean) ;
    imdb.meta.sets = {'train', 'val'} ;
end

end %%end function



% -------------------------------------------------------------------------------------------------
function [rect] = get_axis_aligned_BB(region)
%GETAXISALIGNEDBB computes axis-aligned bbox with same area as the rotated one (REGION)
% -------------------------------------------------------------------------------------------------

cx = mean(region(:,1:2:end),2);
cy = mean(region(:,2:2:end),2);
x1 = min(region(:,1:2:end),[],2);
x2 = max(region(:,1:2:end),[],2);
y1 = min(region(:,2:2:end),[],2);
y2 = max(region(:,2:2:end),[],2);
x1y1x2y2 = region(:,1:2) - region(:,3:4);
x2y2x3y3 = region(:,3:4) - region(:,5:6);
A1 = sqrt(sum(x1y1x2y2.*x1y1x2y2,2)).* sqrt(sum(x2y2x3y3.*x2y2x3y3,2));
A2 = (x2 - x1) .* (y2 - y1);
s = sqrt(A1./A2);
w = s .* (x2 - x1) + 1;
h = s .* (y2 - y1) + 1;
rect = [cx-w/2,cy-h/2,w,h];

% rect = [min(region(:,1:2:end),[],2),...
%         min(region(:,2:2:end),[],2),...
%         max(region(:,1:2:end),[],2)-...
%         min(region(:,1:2:end),[],2),...
%         max(region(:,2:2:end),[],2)-...
%         min(region(:,2:2:end),[],2)];

end
