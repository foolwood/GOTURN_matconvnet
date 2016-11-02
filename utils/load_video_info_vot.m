function [img_files, ground_truth] = load_video_info_vot(base_path, video)

if base_path(end) ~= '/' && base_path(end) ~= '\',
    base_path(end+1) = '/';
end
video_path = [base_path video '/'];

filename = [video_path 'groundtruth.txt'];
ground_truth = csvread(filename);

if(size(ground_truth,2) == 4)
    ground_truth = [ground_truth(:,1),ground_truth(:,2),...
        ground_truth(:,1),ground_truth(:,2)+ground_truth(:,4),...
        ground_truth(:,1)+ground_truth(:,3),ground_truth(:,2)+ground_truth(:,4),...
        ground_truth(:,1)+ground_truth(:,3),ground_truth(:,2)];
end

img_files = dir([video_path '*.jpg']);
if isempty(img_files),
    img_files = dir([video_path '*.png']);
    assert(~isempty(img_files), 'No image files to load.')
end
img_files = sort({img_files.name});
img_files = fullfile(video_path,img_files);
end

