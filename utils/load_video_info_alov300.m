function [img_files, ground_truth] = load_video_info_alov300(base_path, video)

filename = fullfile(base_path,'alov300++_rectangleAnnotation_full',video(1:end-11),[video '.ann']);
ground_truth = dlmread(filename);
frame_gt_index = ground_truth(:,1);
frame_start_index = frame_gt_index(1);
frame_end_index = frame_gt_index(end);
frame_index = (frame_start_index:frame_end_index)';

x_min = min(ground_truth(:,2:2:end),[],2);
y_min = min(ground_truth(:,3:2:end),[],2);
x_max = max(ground_truth(:,2:2:end),[],2);
y_max = max(ground_truth(:,3:2:end),[],2);
x_min = interp1(frame_gt_index,x_min,frame_index);
y_min = interp1(frame_gt_index,y_min,frame_index);
x_max = interp1(frame_gt_index,x_max,frame_index);
y_max = interp1(frame_gt_index,y_max,frame_index);
ground_truth = [x_min,y_min,x_min,y_max,x_max,y_max,x_max,y_min];

image_path = fullfile(base_path,'imagedata++',video(1:end-11),video);
img_files = num2str((frame_index), '%08i.jpg');
img_files = cellstr(img_files);
img_files = fullfile(image_path,img_files);
end

