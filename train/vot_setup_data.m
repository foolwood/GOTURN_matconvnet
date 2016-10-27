function imdb = vot_setup_data(varargin)

% opts = vl_argparse(opts, varargin) ;

opts = [];
opts.size = [227,227];

opts.dataDir = '../data';
opts = vl_argparse(opts, varargin) ;
opts.train_dataDir = fullfile(opts.dataDir,'VOT15');
opts.val_dataDir = fullfile(opts.dataDir,'VOT14');

opts.show_visualization = false;
opts.make_imdb = ispc();%make sure it will take a looooong time & memory

imdb = [];
% -------------------------------------------------------------------------
%                                                             Training Data
% -------------------------------------------------------------------------


if(opts.make_imdb)%vot15:21395 vot14:10188
    set = [ones(1,21395) 2*ones(1,10188)];
    imdb.images.target = zeros(opts.size(1),opts.size(2),3,...
        numel(set),'single');
    imdb.images.search = zeros(opts.size(1),opts.size(2),3,...
        numel(set),'single');
    imdb.images.bboxs = zeros(1,1,4,numel(set),'single');
end

now_index = 1;
bboxs113 = zeros(1,1,4,'single');

disp('Training Data:');
dirs = dir(opts.train_dataDir);
videos = {dirs.name};
videos(strcmp('.', videos) | strcmp('..', videos)| ~[dirs.isdir]) = [];
for  v = 1:numel(videos)
    video = videos{v};disp(video);
    
    [img_files, ground_truth_4xy] = load_video_info_vot(opts.train_dataDir, video);
    ground_truth_rect = [min(ground_truth_4xy(:,1:2:end),[],2),...
                         min(ground_truth_4xy(:,2:2:end),[],2),...
                         max(ground_truth_4xy(:,1:2:end),[],2)-...
                         min(ground_truth_4xy(:,1:2:end),[],2),...
                         max(ground_truth_4xy(:,2:2:end),[],2)-...
                         min(ground_truth_4xy(:,2:2:end),[],2)];
    raw_img_bank = vl_imreadjpeg(img_files);
    im_cols = size(raw_img_bank{1},2);
    im_rows = size(raw_img_bank{1},1);
    for frame = 1:(numel(img_files)-1)
        raw_im_target = raw_img_bank{frame};
        raw_im_search = raw_img_bank{frame+1};
        
        rect_to_crop = ground_truth_rect(frame,:);
        rect_result = ground_truth_rect(frame+1,:);
        
        ltrb_to_crop2 = [rect_to_crop(1)-rect_to_crop(3)/2,...
                         rect_to_crop(2)-rect_to_crop(4)/2,...
                         rect_to_crop(1)+rect_to_crop(3)*3/2,...
                         rect_to_crop(2)+rect_to_crop(4)*3/2];
        ltrb_result = [rect_result(1:2),rect_result(1:2)+rect_result(3:4)];
        
        xs = round(ltrb_to_crop2(1):ltrb_to_crop2(3));
        ys = round(ltrb_to_crop2(2):ltrb_to_crop2(4));
        
        ltrb_result_in_crop2 = ltrb_result-[xs(1),ys(1),xs(1),ys(1)];
        w = numel(xs);h = numel(ys);
        ltrb_result_in_im_search = (ltrb_result_in_crop2.*...
            [opts.size,opts.size]./[w,h,w,h]);
        
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
            rectangle('Position',[ltrb_result_in_im_search(1:2),...
                ltrb_result_in_im_search(3:4)-ltrb_result_in_im_search(1:2)]);
            hold off ;
            
            subplot(2,2,3);imshow(uint8(crop_im_target));
            subplot(2,2,4);imshow(uint8(crop_im_search));
            hold on ;
            rectangle('Position',[ltrb_result_in_crop2(1:2),...
                ltrb_result_in_crop2(3:4)-ltrb_result_in_crop2(1:2)]);
            hold off ;
            drawnow;
        end
        
        if(opts.make_imdb)
            imdb.images.target(:,:,:,now_index) = single(im_target);
            imdb.images.search(:,:,:,now_index) = single(im_search);
            bboxs113(1:4) = ltrb_result_in_im_search(1:4);
            imdb.images.bboxs(:,:,:,now_index) = single(bboxs113) ;
            now_index = now_index+1;
        end
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
    video = videos{v};
    [img_files, ground_truth_4xy] = load_video_info_vot(opts.val_dataDir, video);
    ground_truth_rect = [min(ground_truth_4xy(:,1:2:end),[],2),...
                         min(ground_truth_4xy(:,2:2:end),[],2),...
                         max(ground_truth_4xy(:,1:2:end),[],2)-...
                         min(ground_truth_4xy(:,1:2:end),[],2),...
                         max(ground_truth_4xy(:,2:2:end),[],2)-...
                         min(ground_truth_4xy(:,2:2:end),[],2)];
    raw_img_bank = vl_imreadjpeg(img_files);
    im_cols = size(raw_img_bank{1},2);
    im_rows = size(raw_img_bank{1},1);
    for frame = 1:(numel(img_files)-1)
        raw_im_target = raw_img_bank{frame};
        raw_im_search = raw_img_bank{frame+1};
        
        rect_to_crop = ground_truth_rect(frame,:);
        rect_result = ground_truth_rect(frame+1,:);
        
        ltrb_to_crop2 = [rect_to_crop(1)-rect_to_crop(3)/2,...
                         rect_to_crop(2)-rect_to_crop(4)/2,...
                         rect_to_crop(1)+rect_to_crop(3)*3/2,...
                         rect_to_crop(2)+rect_to_crop(4)*3/2];
        ltrb_result = [rect_result(1:2),rect_result(1:2)+rect_result(3:4)];
        
        xs = round(ltrb_to_crop2(1):ltrb_to_crop2(3));
        ys = round(ltrb_to_crop2(2):ltrb_to_crop2(4));
        
        ltrb_result_in_crop2 = ltrb_result-[xs(1),ys(1),xs(1),ys(1)];
        w = numel(xs);h = numel(ys);
        ltrb_result_in_im_search = (ltrb_result_in_crop2.*...
            [opts.size,opts.size]./[w,h,w,h]);
        
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
            rectangle('Position',[ltrb_result_in_im_search(1:2),...
                ltrb_result_in_im_search(3:4)-ltrb_result_in_im_search(1:2)]);
            hold off ;
            
            subplot(2,2,3);imshow(uint8(crop_im_target));
            subplot(2,2,4);imshow(uint8(crop_im_search));
            hold on ;
            rectangle('Position',[ltrb_result_in_crop2(1:2),...
                ltrb_result_in_crop2(3:4)-ltrb_result_in_crop2(1:2)]);
            hold off ;
            drawnow;
        end
        
        if(opts.make_imdb)
            imdb.images.target(:,:,:,now_index) = single(im_target);
            imdb.images.search(:,:,:,now_index) = single(im_search);
            bboxs113(1:4) = ltrb_result_in_im_search(1:4);
            imdb.images.bboxs(:,:,:,now_index) = single(bboxs113) ;
            now_index = now_index+1;
        end
    end %%end frame
end %%end v

if(opts.make_imdb)
    dataMean = mean(imdb.images.target(:,:,:,set == 1), 4);
    imdb.images.data_mean = dataMean;
    imdb.meta.sets = {'train', 'val'} ;
end

end %%end function