function [img_files, ground_truth] = load_video_info_vid(base_path, video)

if base_path(end) ~= '/' && base_path(end) ~= '\',
    base_path(end+1) = '/';
end
video_path = [base_path video '/'];


if numel(strfind(video,'train'))~=0
    filename = fullfile(base_path,'..','..','..','Annotations/VID/train/',video,'*.xml');
else
    filename = fullfile(base_path,'..','..','..','Annotations/VID/val/',video,'*.xml');
end
for i = 1:numel(filename)
    
    
    
end
filename = [video_path 'groundtruth.txt'];
ground_truth = csvread(filename);

if(size(ground_truth,2) == 4)
    ground_truth = [ground_truth(:,1),ground_truth(:,2),...
        ground_truth(:,1),ground_truth(:,2)+ground_truth(:,4),...
        ground_truth(:,1)+ground_truth(:,3),ground_truth(:,2)+ground_truth(:,4),...
        ground_truth(:,1)+ground_truth(:,3),ground_truth(:,2)];
end

img_files = dir([video_path '*.jpeg']);

img_files = sort({img_files.name});
img_files = fullfile(video_path,img_files);
end

