function [img_files, ground_truth] = load_video_info_vot(base_path, video)

	if base_path(end) ~= '/' && base_path(end) ~= '\',
		base_path(end+1) = '/';
	end
	video_path = [base_path video '/'];

	filename = [video_path 'groundtruth.txt'];
	f = fopen(filename);
	assert(f ~= -1, ['No initial position or ground truth to load ("' filename '").'])
	
	%the format is [x1, y1, x2, y2, x3, y3, x4, y4]
	try
		ground_truth = textscan(f, '%f,%f,%f,%f,%f,%f,%f,%f', 'ReturnOnError',false);  
	catch  %try different format (no commas)
		frewind(f);
		ground_truth = textscan(f, '%f %f %f %f %f %f %f %f');  
	end
	ground_truth = cat(2, ground_truth{:});
	fclose(f);
	
	
	img_files = dir([video_path '*.jpg']);
    if isempty(img_files),
        img_files = dir([video_path '*.png']);
        assert(~isempty(img_files), 'No image files to load.')
    end
    img_files = sort({img_files.name});
	img_files = fullfile(video_path,img_files);
end

