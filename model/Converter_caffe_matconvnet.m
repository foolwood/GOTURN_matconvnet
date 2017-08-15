addpath /home/qwang/caffe/matlab
model = './tracker.prototxt';
weights = './tracker.caffemodel';

caffe.set_mode_cpu();
net = caffe.Net(model, weights, 'test'); % create net and load weights

conv1f = net.params('conv1', 1).get_data();
conv1b = net.params('conv1', 2).get_data();

conv2f = net.params('conv2', 1).get_data();
conv2b = net.params('conv2', 2).get_data();

conv3f = net.params('conv3', 1).get_data();
conv3b = net.params('conv3', 2).get_data();

conv4f = net.params('conv4', 1).get_data();
conv4b = net.params('conv4', 2).get_data();

conv5f = net.params('conv5', 1).get_data();
conv5b = net.params('conv5', 2).get_data();

conv1f_p = net.params('conv1_p', 1).get_data();
conv1b_p = net.params('conv1_p', 2).get_data();

conv2f_p = net.params('conv2_p', 1).get_data();
conv2b_p = net.params('conv2_p', 2).get_data();

conv3f_p = net.params('conv3_p', 1).get_data();
conv3b_p = net.params('conv3_p', 2).get_data();

conv4f_p = net.params('conv4_p', 1).get_data();
conv4b_p = net.params('conv4_p', 2).get_data();

conv5f_p = net.params('conv5_p', 1).get_data();
conv5b_p = net.params('conv5_p', 2).get_data();


conv6f = net.params('fc6-new', 1).get_data();
conv6b = net.params('fc6-new', 2).get_data();

conv7f = net.params('fc7-new', 1).get_data();
conv7b = net.params('fc7-new', 2).get_data();

conv7fb = net.params('fc7-newb', 1).get_data();
conv7bb = net.params('fc7-newb', 2).get_data();

conv8f = net.params('fc8-shapes', 1).get_data();
conv8b = net.params('fc8-shapes', 2).get_data();

save('./GOTURN-Net-pram.mat',...
    'conv1f','conv1b','conv2f','conv2b','conv3f','conv3b','conv4f','conv4b','conv5f','conv5b',...
    'conv1f_p','conv1b_p','conv2f_p','conv2b_p','conv3f_p','conv3b_p','conv4f_p','conv4b_p','conv5f_p','conv5b_p',...
    'conv6f','conv6b','conv7f','conv7b','conv7fb','conv7bb','conv8f','conv8b');

