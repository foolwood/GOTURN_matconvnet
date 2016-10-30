function [img_files, ground_truth] = load_video_info_vot(base_path, video)

	if base_path(end) ~= '/' && base_path(end) ~= '\',
		base_path(end+1) = '/';
	end
	video_path = [base_path video '/'];

	filename = [video_path 'groundtruth.txt'];
    ground_truth = csvread(filename);
	
	
	img_files = dir([video_path '*.jpg']);
    if isempty(img_files),
        img_files = dir([video_path '*.png']);
        assert(~isempty(img_files), 'No image files to load.')
    end
    img_files = sort({img_files.name});
	img_files = fullfile(video_path,img_files);
end

