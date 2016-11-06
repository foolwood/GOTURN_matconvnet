function [img_files, ground_truth, info] = load_video_info_vid(base_path, video)

synsets = {'n02691156','n02419796','n02131653','n02834778','n01503061',...
'n02924116','n02958343','n02402425','n02084071','n02121808','n02503517',...
'n02118333','n02510455','n02342885','n02374451','n02129165','n01674464',...
'n02484322','n03790512','n02324045','n02509815','n02411705','n01726692',...
'n02355227','n02129604','n04468005','n01662784','n04530566','n02062744',...
'n02391049'};
% synnames = {'airplane','antelope','bear','bicycle','bird','bus','car',...
%     'cattle','dog','domestic_cat','elephant','fox','giant_panda',...
%     'hamster','horse','lion','lizard','monkey','motorcycle','rabbit',...
%     'red_panda','sheep','snake','squirrel','tiger','train','turtle',...
%     'watercraft','whale','zebra'};
hash = java.util.Hashtable;
for i = 1:numel(synsets)
    hash.put(synsets{i}, i);
end

if numel(strfind(video,'train')) ~= 0
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
info = cell(0);

for i = 1:(numel(filename))
    rec = VOCreadxml(filename{i});
    img_files{i} = fullfile(video_path,[rec.annotation.filename,'.JPEG']);
    if i == 1
        image_info = imfinfo(img_files{1});
        frame_sz = [image_info.Width,image_info.Height];
        sz = prod(frame_sz(1:2));
    end
    ground_truth{i} = [];
    info{i}.trackid = [];
    if ~isfield(rec.annotation,'object')
        continue;
    end
    vaild_index = 1;
    for k=1:length(rec.annotation.object)
        obj = rec.annotation.object(k);
        c = get_class2node(hash, obj.name);
        b = obj.bndbox;
        bb = str2double({b.xmin b.ymin b.xmax b.ymax});
        bndbox = [bb(1), bb(2), bb(3) - bb(1), bb(4) - bb(2)];
        occluded = str2double(obj.occluded);
        
        if(c ~= 17 && c ~= 23 && c ~= 26 && c ~= 29 && (~occluded) &&...
                sqrt(prod(bndbox(3:4))/sz) < 0.7 &&...
                sqrt(prod(bndbox(3:4))/sz) > 0.1 &&...
                checkBorders(frame_sz,bndbox) && i ~= numel(filename)),
            info{i}.trackid(vaild_index) = str2double(rec.annotation.object(k).trackid);
            info{i}.class(vaild_index) = c;
            info{i}.occluded(vaild_index) = occluded;
            info{i}.bbox(vaild_index,:) = bb-1;%zero-index
            vaild_index = vaild_index+1;
        end 
        ground_truth{i}(k,:) = [bb(1),bb(2),bb(1),bb(4),bb(3),bb(4),bb(3),bb(2)];
    end
end

end

function ok = checkBorders(frame_sz, object_extent)
    dist_from_border = 0.05 * (object_extent(3) + object_extent(4))/2;
    ok = object_extent(1) > dist_from_border && object_extent(2) > dist_from_border && ...
                (frame_sz(1)-(object_extent(1)+object_extent(3))) > dist_from_border && ...
                (frame_sz(2)-(object_extent(2)+object_extent(4))) > dist_from_border;
end

