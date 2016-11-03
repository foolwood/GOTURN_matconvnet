function net = vgg16_net_init()
% input:
%       -target :227*227*3
%       -image :227*227*3
% output:
%       -bbox :4*1*1

run vl_setupnn.m ;
net = dagnn.DagNN() ;
%% TODO

%% save

