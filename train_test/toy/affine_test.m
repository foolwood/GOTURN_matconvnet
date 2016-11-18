bbox2rect = @(x) [x(1:2),x(3:4)-x(1:2)]+1;
scaled2rect = @(x) [x(1),x(2),x(3)-x(1),x(4)-x(2)]/10*227+1;

image_target = imread('00000001.jpg');
image_search = imread('00000002.jpg');
bbox_target = [292.23,128.36,438.19,260.83]-1;%zero-index
bbox_search = [254,110.38,399.31,248.34]-1;%zero-index

subplot(2,2,1);imshow(image_target);rectangle('Position',bbox2rect(bbox_target),'EdgeColor',[0 1 0]);
subplot(2,2,2);imshow(image_search);rectangle('Position',bbox2rect(bbox_search),'EdgeColor',[0 1 0]);
Wo = 227;
Ho = 227;
No = 2;

net = dagnn.DagNN();
SampleGenerator = dagnn.SampleGenerator('Ho',Ho,'Wo',Wo,'No',No);

net.addLayer('SampleGenerator',SampleGenerator,...
    {'bbox_target','bbox_search','image_target','image_search'},...
    {'bbox_gt_scaled','image_target_crop','image_search_crop'});

bbox_target = single(reshape(bbox_target,[1,1,4,1]));
bbox_search = single(reshape(bbox_search,[1,1,4,1]));

net.eval({'bbox_target',bbox_target,'bbox_search',bbox_search,...
    'image_target',single(image_target),'image_search',single(image_search)});

image_target_crop = net.vars(net.getVarIndex('image_target_crop')).value;
image_search_crop = net.vars(net.getVarIndex('image_search_crop')).value;
bbox_gt_scaled = net.vars(net.getVarIndex('bbox_gt_scaled')).value;

subplot(2,2,3);imshow(uint8(image_target_crop(:,:,:,1)));
subplot(2,2,4);imshow(uint8(image_search_crop(:,:,:,2)));rectangle('Position',scaled2rect(bbox_gt_scaled(1,1,1:4,2)),'EdgeColor',[0 1 0]);
