function [img_files, ground_truth] = load_video_info_alov300(base_path, video)

if base_path(end) ~= '/' && base_path(end) ~= '\',
    base_path(end+1) = '/';
end

filename = [base_path 'alov300++_rectangleAnnotation_full/' video(1:end-11) '/' video '.ann'];
ground_truth = dlmread(filename);
frame_index = ground_truth(:,1);

x_min = min(ground_truth(:,2:2:end),[],2);
y_min = min(ground_truth(:,3:2:end),[],2);
x_max = max(ground_truth(:,2:2:end),[],2);
y_max = max(ground_truth(:,3:2:end),[],2);
ground_truth = [x_min,y_min,x_max,y_max];

image_path = [base_path 'imagedata++/' video(1:end-11) '/' video '/'];
img_files = dir([image_path '*.jpg']);
if isempty(img_files),
    img_files = dir([image_path '*.png']);
    assert(~isempty(img_files), 'No image files to load.')
end
img_files = sort({img_files.name});
img_files = fullfile(video_path,img_files);
end

