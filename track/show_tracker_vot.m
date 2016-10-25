function show_tracker_vot(net_file,base_path,gpu_id,start_vidoe_num,show_visualization)

if nargin < 1, net_file = '../model/GOTURN_net.mat'; end
if nargin < 2, base_path = '../VOT15/'; end
if nargin < 3, gpu_id = []; end
if nargin < 4, start_vidoe_num = 1; end
if nargin < 5, show_visualization = 1; end

if exist(net_file,'file')
    net = dagnn.DagNN.loadobj(load(net_file));
else
    error('No net to load!!!!');
end
% if numel(gpu_id)>0    %TODO
%    gpuDevice(gpu_id) ;
%    net.move('gpu') ; 
% end
net.mode = 'test';

dirs = dir(base_path);
videos = {dirs.name};
videos(strcmp('.', videos) | strcmp('..', videos)| ~[dirs.isdir]) = [];

for  v = start_vidoe_num:1%numel(videos)
    video = videos{v};
    [img_files, ground_truth] = load_video_info_vot(base_path, video);
    [result, time] = tracker(img_files, ground_truth, net, gpu_id, show_visualization);
    fprintf('Video: %d , fps:%3.3f\n',v,size(result,1)/time);
end

end %%function

