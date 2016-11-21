function affine6_test()

image_target = imread('00000001.jpg');
image_search = imread('00000002.jpg');
xy4_target = [334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41];%zero-index
xy4_search = [399.31,186.3,350.5,248.34,254,172.42,302.81,110.38];%zero-index

subplot(2,4,1);imshow(image_target);hold on;
plot(xy4_target([1:2:end,1]),xy4_target([2:2:end,2]),'g');title('target image')
subplot(2,4,2);imshow(image_search);hold on;
plot(xy4_search([1:2:end,1]),xy4_search([2:2:end,2]),'g');title('search image')

% [tform1] = ...
%     estimateGeometricTransform(reshape(xy4_target,2,[])',[1,227,227,1;1,1,227,227]',...
%     'affine');
% outputView = imref2d([227,227]);
% Ir = imwarp(image_target,tform1,'OutputView',outputView);
% subplot(2,4,3); imshow(Ir);
% 
% [tform2] = ...
%     estimateGeometricTransform(reshape(xy4_search,2,[])',[1,227,227,1;1,1,227,227]',...
%     'affine');
% outputView = imref2d([227,227]);
% Ir = imwarp(image_search,tform2,'OutputView',outputView);
% subplot(2,4,4); imshow(Ir);

Wo = 227;
Ho = 227;
No = 2;
im_sz = size(image_target);



net = dagnn.DagNN();
AffineGridGenerator = dagnn.AffineGridGenerator('Ho',Ho,'Wo',Wo);
net.addLayer('AffineGridGenerator',AffineGridGenerator,{'aff'},{'grid'});
sampler = dagnn.BilinearSampler();
net.addLayer('samp',sampler,{'input','grid'},{'output'});

net.eval({'input',single(image_target),'aff',xy4_2_aff6(xy4_target,im_sz,0)});
image_target_crop = net.vars(net.getVarIndex('output')).value;

net.eval({'input',single(image_search),'aff',xy4_2_aff6(xy4_search,im_sz,0)});
image_search_crop = net.vars(net.getVarIndex('output')).value;

subplot(2,3,4);imshow(uint8(image_target_crop(:,:,:,1)));title('tight target')
subplot(2,3,5);imshow(uint8(image_search_crop(:,:,:,1)));title('tight search')

net.eval({'input',single(image_target),'aff',xy4_2_aff6(xy4_target,im_sz,1.5)});
image_target_crop = net.vars(net.getVarIndex('output')).value;

net.eval({'input',single(image_search),'aff',xy4_2_aff6(xy4_target,im_sz,1.5)});
image_search_crop = net.vars(net.getVarIndex('output')).value;

subplot(2,3,3);imshow(uint8(image_target_crop(:,:,:,1)));title('target with surrounding')
subplot(2,3,6);imshow(uint8(image_search_crop(:,:,:,1)));title('search with surrounding')


end


function aff6 = xy4_2_aff6(xy4,im_sz,pad)
xy4 = (((xy4-1)./im_sz([2,1,2,1,2,1,2,1]))*2-1);
% plot([-1,1,1,-1],[-1,-1,1,1],'r*');hold on
% plot(xy4(1:2:end),xy4(2:2:end),'r*');
aff6 = [xy4(8)-xy4(2),xy4(4)-xy4(2),...
    xy4(7)-xy4(1),xy4(3)-xy4(1),...
    xy4(4)+xy4(8),xy4(3)+xy4(7)]/2;

if nargin > 2
   aff6(1:4) = aff6(1:4)*(1+pad);
end
aff6 = single(reshape(aff6,1,1,6,[]));
end