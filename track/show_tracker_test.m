addpath('../utils');
run vl_setupnn;
if ispc()
    net_file = '../model/GOTURN_net.mat';
    base_path = '../data/VOT15/';
    gpu_id = [1,2];
    start_vidoe_num = 1;
    show_visualization = 0;
else
    net_file = '../model/GOTURN_net.mat';
    base_path = '../data/VOT15/';
    gpu_id = [];
    start_vidoe_num = 1;
    show_visualization = 0;
end

show_tracker_vot(net_file,base_path,gpu_id,start_vidoe_num,show_visualization);
