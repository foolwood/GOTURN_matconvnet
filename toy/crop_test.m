ccc
I = imread('street2.jpg');

imwrite(I,'street_test1.jpeg');
imwrite(I,'street_test2.jpg');

I = imread('street_test1.jpeg');
I2 = vl_imreadjpeg({'street_test1.jpeg'});
sum(abs(single(I(:))-I2{1,1}(:))) %%why!!241016

I = imread('street_test2.jpg');
I2 = vl_imreadjpeg({'street_test2.jpg'});
sum(abs(single(I(:))-I2{1,1}(:))) %%why!!241016

I(1,1,:)
J = imcrop(I,[1.3,1.3,3,3])
J = imcrop(I,[1.5,1.5,3,3])



% J_pad = padarray(J,[1,2])

b = padarray([1 2 3 4],3,'symmetric','pre')


B = padarray([1 2; 3 4],[3 2],'replicate','post')

A = [1 2; 3 4];
B = [5 6; 7 8];
C = cat(3,A,B)
D = padarray(C,[3 3],0,'both')