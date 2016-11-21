function [img_files, ground_truth_4xy] = load_video_info_det(det_dataDir)

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
xml_file = cell(0);
for i = 1:numel(train_anno_subfile)
    xml_sub_file = dir(fullfile(train_anno_subfile{i},'*.xml'));
    xml_sub_file = {xml_sub_file.name};
    xml_sub_file = fullfile(train_anno_subfile{i},xml_sub_file);
    xml_file = [xml_file,xml_sub_file];
end

vaild_index = 0;
img_files = cell(1,478806);
ground_truth_4xy = zeros(478806,8,'single');
for i = 1:numel(xml_file)
    rec = VOCreadxml(xml_file{i});
    if ~isfield(rec.annotation,'object')
        continue;
    end
    img_files_temp = strrep(xml_file{i},'Annotations','Data');
    img_files_temp = strrep(img_files_temp,'xml','JPEG');
    for k=1:length(rec.annotation.object)
        
        obj = rec.annotation.object(k);
        b = obj.bndbox;
        bb = str2double({b.xmin b.ymin b.xmax b.ymax});
        
        vaild_index =vaild_index+1;
        
        img_files{vaild_index} = img_files_temp;

        ground_truth_4xy(vaild_index,:) = [bb(1),bb(2),bb(1),bb(4),bb(3),bb(4),bb(3),bb(2)];
        
    end
end
end