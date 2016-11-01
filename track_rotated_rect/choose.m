function choose()
addpath('../utils');
base_path = '../data/VOT15/';
show_visualization = 1;


dirs = dir(base_path);
videos = {dirs.name};
videos(strcmp('.', videos) | strcmp('..', videos)| ~[dirs.isdir]) = [];

v = listdlg('ListString',videos, 'Name','Choose video', 'SelectionMode','single');
if isempty(v),return; end
video = videos{v};

[img_files, ground_truth] = load_video_info_vot(base_path, video);
r_tracker(img_files, ground_truth, show_visualization);

end %%function

