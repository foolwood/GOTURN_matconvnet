function [img_files, ground_truth_4xy,img_display_sz] = load_video_info_det(det_dataDir)

train_anno_path = fullfile(det_dataDir,'Annotations','DET','train');
train2013_anno_subfile = fullfile(train_anno_path,'ILSVRC2013_train');
train2013_anno_subfile = dir(fullfile(train2013_anno_subfile,'n*'));
train2013_anno_subfile = fullfile(train_anno_path,'ILSVRC2013_train',...
    {train2013_anno_subfile.name});
train2014_anno_subfile = fullfile(train_anno_path,{'ILSVRC2014_train_0000',...
    'ILSVRC2014_train_0001','ILSVRC2014_train_0002',...
    'ILSVRC2014_train_0003','ILSVRC2014_train_0004',...
    'ILSVRC2014_train_0005','ILSVRC2014_train_0006'});
train_anno_subfile = [train2013_anno_subfile,train2014_anno_subfile];
xml_file = cell(1,349319);
index_count = 0;
for i = 1:numel(train_anno_subfile)
    xml_sub_file = dir(fullfile(train_anno_subfile{i},'*.xml'));
    xml_sub_file = {xml_sub_file.name};
    xml_sub_file = fullfile(train_anno_subfile{i},xml_sub_file);
    n_xml_sub_file = numel(xml_sub_file);
    xml_file(index_count+(1:n_xml_sub_file)) = xml_sub_file;
    index_count = index_count+n_xml_sub_file;
end

n_xml_file = numel(xml_file);%349319
kMaxRatio = 0.66;
img_files = cell(1,n_xml_file);
ground_truth_4xy = zeros(n_xml_file,8,'single');
img_display_sz = ones(n_xml_file,2,'single');
vaild_index = 0;

for i = 1:n_xml_file
    rec = VOCreadxml(xml_file{i});
    if ~isfield(rec.annotation,'object') || ~isfield(rec.annotation,'size')
        continue;
    end
    img_files_temp = strrep(xml_file{i},'Annotations','Data');
    img_files_temp = strrep(img_files_temp,'xml','JPEG');
    
    display_width = str2double(rec.annotation.size.width);
    display_height = str2double(rec.annotation.size.height);
    
    for k=1:length(rec.annotation.object)
        obj = rec.annotation.object(k);
        b = obj.bndbox;
        bb = str2double({b.xmin b.ymin b.xmax b.ymax});
        w = bb(3)-bb(1);
        h = bb(4)-bb(2);
        if (w > kMaxRatio * display_width || bb(1) < 0.5*w || (display_width - bb(3)) < 0.5*w ||...
                h > kMaxRatio * display_height || bb(2) < 0.5*h || (display_height - bb(4)) < 0.5*h ||...
                bb(1) < 0 || bb(2) < 0 || bb(3) <= bb(1) || bb(4) <= bb(2))
            continue;
        end
        gt_4xy_temp = single([bb(1),bb(2),bb(3),bb(2),bb(3),bb(4),bb(1),bb(4)]+1);
        
        vaild_index = vaild_index + 1;
        img_files{vaild_index} = img_files_temp;
        ground_truth_4xy(vaild_index,:) = gt_4xy_temp;
        img_display_sz(vaild_index,:) = [display_width,display_height];
    end
end
img_files((vaild_index+1):n_xml_file) = [];
ground_truth_4xy((vaild_index+1):n_xml_file,:) = [];
img_display_sz((vaild_index+1):n_xml_file,:) = [];
end