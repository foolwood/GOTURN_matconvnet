function [img_files, ground_truth] = load_video_info_vid(base_path, video)
if base_path(end) ~= '/' && base_path(end) ~= '\',
    base_path(end+1) = '/';
end
if numel(strfind(video,'train'))~=0
    filename = dir(fullfile(base_path,'Annotations','VID','train',video,'*.xml'));
    filename = sort({filename.name});
    filename = fullfile(base_path,'Annotations','VID','train',video,filename);
    video_path = fullfile(base_path,'Data','VID','train',video);
else
    filename = dir(fullfile(base_path,'Annotations','VID','val',video,'*.xml'));
    filename = sort({filename.name});
    filename = fullfile(base_path,'Annotations','VID','val',video,filename);
    video_path = fullfile(base_path,'Data','VID','val',video);
end

img_files = cell(0);
ground_truth = cell(0);
for i = 1:numel(filename)
    rec = VOCreadxml(filename{i});
    
    img_files{i} = fullfile(video_path,[rec.annotation.filename,'.JPEG']);
    
    ground_truth{i} = [];
    if ~isfield(rec.annotation,'object')
        continue;
    end
    for k=1:length(rec.annotation.object)
        
        obj = rec.annotation.object(k);
%         c = get_class2node(hash, obj.name);
        b = obj.bndbox;
        bb = str2double({b.xmin b.ymin b.xmax b.ymax});
%         bndbox = [bb(1), bb(2), bb(3) - bb(1), bb(4) - bb(2)];
%         occluded = str2double(obj.occluded);
        ground_truth{i}(end+1,:) = [bb(1),bb(2),bb(1),bb(4),bb(3),bb(4),bb(3),bb(2)];
        
    end
end

end

