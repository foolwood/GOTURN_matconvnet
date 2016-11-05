addpath('../../utils');
addpath('../ILSVRC/devkit/evaluation')
ccc

%% load seqName
VOT16 = importdata('../VOT16/list.txt');
VOT15 = importdata('../VOT15/list.txt');
VOT14 = importdata('../VOT14/list.txt');
VOT13 = importdata('../VOT13/list.txt');

% CVPR2013 = readtable('../OTB/cvpr13.txt','ReadVariableNames',false);
% CVPR2013 = CVPR2013.Var1;
% TB100 = readtable('../OTB/tb_100.txt','ReadVariableNames',false);
% TB100 = TB100.Var1;
% TB50 = readtable('../OTB/tb_50.txt','ReadVariableNames',false);
% TB50 = TB50.Var1;

NUS_PRO_all = importdata('../NUS_PRO/seq_list.csv');
NUS_PRO = importdata('../NUS_PRO/seq_list_with_gt.csv');

TC128_temp = dir('../TC128');
TC128 = {TC128_temp.name};
TC128(strcmp('.', TC128) | strcmp('..', TC128)| ~[TC128_temp.isdir]) = [];

ALOV300_temp1 = dir('../ALOV300/imagedata++');
ALOV300_temp2 = {ALOV300_temp1.name};
ALOV300_temp2(strcmp('.', ALOV300_temp2) | strcmp('..', ALOV300_temp2)| ~[ALOV300_temp1.isdir]) = [];
ALOV300 = cell(0);
for i = 1:numel(ALOV300_temp2)
    ALOV300_temp3 = dir(fullfile('../ALOV300/imagedata++',ALOV300_temp2{i}));
    ALOV300_temp4 = {ALOV300_temp3.name};
    ALOV300_temp4(strcmp('.', ALOV300_temp4) | strcmp('..', ALOV300_temp4)| ~[ALOV300_temp3.isdir]) = [];
    ALOV300(end+(1:numel(ALOV300_temp4))) = ALOV300_temp4;
end


ILSVRC_train_temp1 = dir('../ILSVRC/Data/VID/train/ILSVRC2015_VID_train_*');
ILSVRC_train_temp1 = sort({ILSVRC_train_temp1.name});
ILSVRC_train = cell(0);
for i = 1:numel(ILSVRC_train_temp1)
   ILSVRC_train_temp2 = dir(['../ILSVRC/Data/VID/train/',...
       ILSVRC_train_temp1{i},'/ILSVRC2015_train_*']);
   ILSVRC_train_temp2 = sort({ILSVRC_train_temp2.name});
   ILSVRC_train(end+(1:numel(ILSVRC_train_temp2))) = fullfile(ILSVRC_train_temp1{i},ILSVRC_train_temp2);
end
ILSVRC_val = dir('../ILSVRC/Data/VID/val/ILSVRC2015_val_*');
ILSVRC_val = sort({ILSVRC_val.name});
ILSVRC = [ILSVRC_train,ILSVRC_val];

clear i filename ILSVRC_train_temp ALOV300_temp1 ALOV300_temp2 ALOV300_temp3 ALOV300_temp4 TC128_temp

%% load groundtruth
visual = true;
% base_path = '../VOT16/';
% for i = 1:numel(VOT16)
%     disp(i)
%     video = VOT16{i};
%     [img_files, ground_truth] = load_video_info_vot(base_path, video);
%     if numel(img_files)~=size(ground_truth,1)
%         error('miss_matching');
%     end
%     VOT16_frame_num(i) = numel(img_files);
%     if visual,
%         close all
%         update_visualization = show_video(img_files);
%         for frame = 2:numel(img_files),
%             stop = update_visualization(frame, ground_truth(frame,:),[],[]);
%             if stop, break, end
%             drawnow;
%         end
%     end
% end
% 
% base_path = '../VOT15/';
% for i = 1:numel(VOT15)
%     disp(i)
%     video = VOT15{i};
%     [img_files, ground_truth] = load_video_info_vot(base_path, video);
%     if numel(img_files)~=size(ground_truth,1)
%         error('miss_matching');
%     end
%     VOT15_frame_num(i) = numel(img_files);
%     if visual,
%         close all
%         update_visualization = show_video(img_files);
%         for frame = 2:numel(img_files),
%             stop = update_visualization(frame, ground_truth(frame,:),[],[]);
%             if stop, break, end
%             drawnow;
%         end
%     end
% end

% base_path = '../VOT14/';
% for i = 1:numel(VOT14)
%     disp(i)
%     video = VOT14{i};
%     [img_files, ground_truth] = load_video_info_vot(base_path, video);
%     if numel(img_files)~=size(ground_truth,1)
%         error('miss_matching');
%     end
%     VOT14_frame_num(i) = numel(img_files);
%     if visual,
%         close all
%         update_visualization = show_video(img_files);
%         for frame = 2:numel(img_files),
%             stop = update_visualization(frame, ground_truth(frame,:),[],[]);
%             if stop, break, end
%             drawnow;
%         end
%     end
% end
% 
% base_path = '../VOT13/';
% for i = 1:numel(VOT13)
%     disp(i)
%     video = VOT13{i};
%     [img_files, ground_truth] = load_video_info_vot(base_path, video);
%     if numel(img_files)~=size(ground_truth,1)
%         error('miss_matching');
%     end
%     VOT13_frame_num(i) = numel(img_files);
%     if visual,
%         close all
%         update_visualization = show_video(img_files);
%         for frame = 2:numel(img_files),
%             stop = update_visualization(frame, ground_truth(frame,:),[],[]);
%             if stop, break, end
%             drawnow;
%         end
%     end
% end
% 
% base_path = '../OTB/';
% for i = 1:numel(CVPR2013)
%     disp(i)
%     video = CVPR2013{i};
%     [img_files, ground_truth] = load_video_info_otb(base_path, video);
%     if numel(img_files)~=size(ground_truth,1)
%         error('miss_matching');
%     end
%     CVPR2013_frame_num(i) = numel(img_files);
%     if visual,
%         close all
%         update_visualization = show_video(img_files);
%         for frame = 2:numel(img_files),
%             stop = update_visualization(frame, ground_truth(frame,:),[],[]);
%             if stop, break, end
%             drawnow;
%         end
%     end
% end
% 
% base_path = '../OTB/';
% for i = 1:numel(TB50)
%     disp(i)
%     video = TB50{i};
%     [img_files, ground_truth] = load_video_info_otb(base_path, video);
%     if numel(img_files)~=size(ground_truth,1)
%         error('miss_matching');
%     end
%     TB50_frame_num(i) = numel(img_files);
%     if visual,
%         close all
%         update_visualization = show_video(img_files);
%         for frame = 2:numel(img_files),
%             stop = update_visualization(frame, ground_truth(frame,:),[],[]);
%             if stop, break, end
%             drawnow;
%         end
%     end
% end

% base_path = '../OTB/';
% for i = 1:numel(TB100)
%     disp(i)
%     video = TB100{i};
%     [img_files, ground_truth] = load_video_info_otb(base_path, video);
%     
%     if numel(img_files)~=size(ground_truth,1)
%         error('miss_matching');
%     end
%     TB100_frame_num(i) = numel(img_files);
%     if visual,
%         close all
%         update_visualization = show_video(img_files);
%         for frame = 2:numel(img_files),
%             stop = update_visualization(frame, ground_truth(frame,:),[],[]);
%             if stop, break, end
%             drawnow;
%         end
%     end
% end
% 
% base_path = '../NUS_PRO/';
% for i = 1:numel(NUS_PRO)
%     disp(i)
%     video = NUS_PRO{i};
%     [img_files, ground_truth] = load_video_info_nus_pro(base_path, video);
%     if numel(img_files)~=size(ground_truth,1)
%         error('miss_matching');
%     end
%     NUS_PRO_frame_num(i) = numel(img_files);
%     if visual,
%         close all
%         update_visualization = show_video(img_files);
%         for frame = 2:numel(img_files),
%             stop = update_visualization(frame, ground_truth(frame,:),[],[]);
%             if stop, break, end
%             drawnow;
%         end
%     end
% end
% 
% base_path = '../TC128/';
% for i = 1:numel(TC128)
%     disp(i)
%     video = TC128{i};
%     [img_files, ground_truth] = load_video_info_tc128(base_path, video);
%     if numel(img_files)~=size(ground_truth,1)
%         error('miss_matching');
%     end
%     TC128_frame_num(i) = numel(img_files);
%     if visual,
%         close all
%         update_visualization = show_video(img_files);
%         for frame = 2:numel(img_files),
%             stop = update_visualization(frame, ground_truth(frame,:),[],[]);
%             if stop, break, end
%             drawnow;
%         end
%     end
% end
% 
% base_path = '../ALOV300/';
% for i = 1:numel(ALOV300)
%     disp(i)
%     video = ALOV300{i};
%     [img_files, ground_truth] = load_video_info_alov300(base_path, video);
%     if numel(img_files)~=size(ground_truth,1)
%         error('miss_matching');
%     end
%     ALOV300_frame_num(i) = numel(img_files);
%     if visual,
%         close all
%         update_visualization = show_video(img_files);
%         for frame = 1:numel(img_files),
%             stop = update_visualization(frame, ground_truth(frame,:),[],[]);
%             if stop, break, end
%             drawnow;
%         end
%     end
% end

base_path = '../ILSVRC/';
for i = 1:numel(ILSVRC_train)
    disp(i)
    video = ILSVRC_train{i};
    [img_files, ground_truth] = load_video_info_vid(base_path, video);
    if numel(img_files)~=numel(ground_truth)
        error('miss_matching');
    end
    ILSVRC_train_frame_num(i) = numel(img_files);
    if visual,
        close all
        update_visualization = show_video(img_files);
        for frame = 1:numel(img_files),
            stop = update_visualization(frame, [],[],ground_truth{frame});
            if stop, break, end
            drawnow;
        end
    end
end

base_path = '../ILSVRC/';
for i = 1:numel(ILSVRC_val)
    disp(i)
    video = ILSVRC_val{i};
    [img_files, ground_truth] = load_video_info_vid(base_path, video);
    if numel(img_files)~=numel(ground_truth)
        error('miss_matching');
    end
    ILSVRC_train_frame_num(i) = numel(img_files);
    if visual,
        close all
        update_visualization = show_video(img_files);
        for frame = 1:numel(img_files),
            stop = update_visualization(frame, [],[],ground_truth{frame});
            if stop, break, end
            drawnow;
        end
    end
end



