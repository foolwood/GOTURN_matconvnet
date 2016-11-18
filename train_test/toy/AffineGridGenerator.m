function AffineGridGenerator()
run vl_setupnn.m;
input = imread('peppers.png');
image_sz = size(input);image_sz(3) = [];
target_sz = [125,125];


target_aff = [];
target_us = [1,0,0;0.5,0,0]';
target_us = reshape(target_us,1,1,3,[]);

nn = dagnn.DagNN();
scale_grid = dagnn.UniformScalingGridGenerator('Ho',target_sz(1),'Wo',target_sz(2));
nn.addLayer('us', scale_grid,{'ss'},{'grid'});

% aff_grid = dagnn.AffineGridGenerator('Ho',target_sz(1),'Wo',target_sz(2));
% nn.addLayer('aff', aff_grid,{'aff'},{'grid'});

sampler = dagnn.BilinearSampler();
nn.addLayer('samp',sampler,{'input','grid'},{'output'});

nn.eval({'input',single(input),'ss',single(target_us)});
output = nn.vars(nn.getVarIndex('output')).value;
subplot(2,2,1);imshow(uint8(output(:,:,:,1)));
subplot(2,2,2);imshow(uint8(output(:,:,:,2)));
c1 = uint8(output(:,:,:,1));
c2 = uint8(output(:,:,:,2));


% xi = linspace(-1, 1, target_sz(1));
% yi = linspace(-1, 1, target_sz(2));
% [yy,xx] = meshgrid(xi,yi);
% xxyy = [xx(:), yy(:)]' ;

% nbatch = size(target_rect_raw,1);
% 
% 
% w = target_rect_raw(:,3);
% h = target_rect_raw(:,4);
% cx = target_rect_raw(:,1)+(w-1)/2;
% cy = target_rect_raw(:,2)+(h-1)/2;
% w_im = image_sz(2);
% h_im = image_sz(1);
% 
% 
% % cy = [100;1];
% % cx = [100;1];
% cy_ = ((cy-1)*2/(h_im-1))-1;
% cx_ = ((cx-1)*2/(w_im-1))-1;
% cyx_ = [cy_,cx_]';
% cyx_ = reshape(cyx_,2,1,[]);
% 
% h_ = h/h_im;
% w_ = w/w_im;
% hw_ =[h_,w_]';
% hw_ = reshape(hw_,2,1,[]);
% 
% 
% g = bsxfun(@times, xxyy, hw_); % scale
% g = bsxfun(@plus, g, cyx_); % translate

% plot(xxyy(1,:),xxyy(2,:),'r*');hold on ;plot(g(1,:,1),g(2,:,1),'b.');
% g = single(reshape(g, 2,target_sz(1),target_sz(2),nbatch));

% nn.eval({'input',single(input),'grid',g});
% output = nn.vars(nn.getVarIndex('output')).value;
% subplot(2,2,3);imshow(uint8(output(:,:,:,1)));
% subplot(2,2,4);imshow(uint8(output(:,:,:,2)));
% c3 = uint8(output(:,:,:,1));
% c4 = uint8(output(:,:,:,2));
% xywh = target_rect_raw(2,:);
% xywh(3:4) = xywh(3:4)-1;
% c5 = imcrop(single(input),xywh);
% sum(c1(:)-c3(:))

target_rect_raw = [200,1,512,384;1,1,125,125];%x,y,w,h
T = rect2scaling_factor(target_rect_raw,image_sz);

nn = dagnn.DagNN();
scale_grid = dagnn.SpecialScalingGridGenerator('Ho',target_sz(1),'Wo',target_sz(2));
nn.addLayer('us', scale_grid,{'T'},{'grid'});
sampler = dagnn.BilinearSampler();
nn.addLayer('samp',sampler,{'input','grid'},{'output'});

nn.eval({'input',single(input),'T',single(T)});
output = nn.vars(nn.getVarIndex('output')).value;
subplot(2,2,3);imshow(uint8(output(:,:,:,1)));
subplot(2,2,4);imshow(uint8(output(:,:,:,2)));


target_rect_raw = [1,1,512,384;1,1,125,125];%x,y,w,h

g = rect2grid(target_rect_raw,image_sz,125,125);
outputs = vl_nnbilinearsampler(single(input), single(g));

subplot(2,2,3);imshow(uint8(outputs(:,:,:,1)));
subplot(2,2,4);imshow(uint8(outputs(:,:,:,2)));

end

function T = rect2scaling_factor(rect,im_sz)
%rect n*4            [xywh]
%T    1*1*4*n        [ssxy]
w = rect(:,3);
h = rect(:,4);
cx = rect(:,1)+(w-1)/2;
cy = rect(:,2)+(h-1)/2;
w_im = im_sz(2);
h_im = im_sz(1);

cy_ = ((cy-1)*2/(h_im-1))-1;
cx_ = ((cx-1)*2/(w_im-1))-1;

h_ = h/h_im;
w_ = w/w_im;
T = [h_,w_,cy_,cx_]';
T = reshape(T,1,1,4,[]);
end

function g = rect2grid(rect,im_sz,Ho , Wo)
%rect n*4            [xywh]
% g   2 x Ho x Wo x N
w = rect(:,3);
h = rect(:,4);
cx = rect(:,1)+(w-1)/2;
cy = rect(:,2)+(h-1)/2;
w_im = im_sz(2);
h_im = im_sz(1);

cy_t = ((cy-1)*2/(h_im-1))-1;
cx_t = ((cx-1)*2/(w_im-1))-1;

h_s = h/h_im;
w_s = w/w_im;

S = reshape([h_s,w_s]', 2,1,[]); % x,y scaling
t = reshape([cy_t,cx_t]', 2,1,[]); % translation

useGPU = isa(t, 'gpuArray');
xi = linspace(-1, 1, Wo);
yi = linspace(-1, 1, Ho);
[yy,xx] = meshgrid(xi,yi);
xxyy = [xx(:), yy(:)]' ;
if useGPU,
    xxyy = gpuArray(xxyy);
end
% transform the grid:
g = bsxfun(@times, xxyy, S); % scale
g = bsxfun(@plus, g, t); % translate
g = reshape(g, 2,Ho,Wo,[]);


end