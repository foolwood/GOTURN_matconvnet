function net = goturn_net_init()
% input:
%       -target :H*W*3
%       -image :H*W*3
% output:
%       -bbox :4*1*1

run vl_setupnn.m;
net = dagnn.DagNN();
net.meta.normalization.averageImage = reshape(single([123,117,104]),[1,1,3]);

SampleGenerator = dagnn.SampleGenerator('Ho',227,'Wo',227,'kGeneratedExamplesPerImage',10,'padding',1,...
    'averageImage',net.meta.normalization.averageImage,'visual',ismac());
net.addLayer('SampleGenerator',SampleGenerator,...
    {'image_prev','image_curr','bbox_prev','bbox_curr'},...
    {'targets','images','bboxes_gt_scaled'});

%% target

conv1 = dagnn.Conv('size', [11 11 3 96], 'pad', 0, 'stride', 4, 'hasBias', true);
net.addLayer('conv1', conv1, {'targets'}, {'conv1'}, {'conv1f', 'conv1b'});
net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
norm1 = dagnn.LRN('param', [5 1 0.0001/5 0.75]);
net.addLayer('norm1', norm1, {'conv1x'}, {'norm1'});
pool1 = dagnn.Pooling('method', 'max', 'poolSize', [3 3],'pad', 0, 'stride', 2);
net.addLayer('pool1', pool1, {'norm1'}, {'pool1'});

conv2 = dagnn.Conv('size', [5 5 48 256], 'pad', 2, 'stride', 1, 'hasBias', true);
net.addLayer('conv2', conv2, {'pool1'}, {'conv2'}, {'conv2f', 'conv2b'});
net.addLayer('relu2', dagnn.ReLU(), {'conv2'}, {'conv2x'});
norm2 = dagnn.LRN('param', [5 1 0.0001/5 0.75]);
net.addLayer('norm2', norm2, {'conv2x'}, {'norm2'});
pool2 = dagnn.Pooling('method', 'max', 'poolSize', [3 3],'pad', 0, 'stride', 2);
net.addLayer('pool2', pool2, {'norm2'}, {'pool2'});

conv3 = dagnn.Conv('size', [3 3 256 384], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv3', conv3, {'pool2'}, {'conv3'}, {'conv3f', 'conv3b'});
net.addLayer('relu3', dagnn.ReLU(), {'conv3'}, {'conv3x'});

conv4 = dagnn.Conv('size', [3 3 192 384], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv4', conv4, {'conv3x'}, {'conv4'}, {'conv4f', 'conv4b'});
net.addLayer('relu4', dagnn.ReLU(), {'conv4'}, {'conv4x'});

conv5 = dagnn.Conv('size', [3 3 192 256], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv5', conv5, {'conv4x'}, {'conv5'}, {'conv5f', 'conv5b'});
net.addLayer('relu5', dagnn.ReLU(), {'conv5'}, {'conv5x'});
pool5 = dagnn.Pooling('method', 'max', 'poolSize', [3 3], 'pad', 0, 'stride', 2);
net.addLayer('pool5', pool5, {'conv5x'}, {'pool5'});

%% image

conv1_p = dagnn.Conv('size', [11 11 3 96], 'pad', 0, 'stride', 4, 'hasBias', true);
net.addLayer('conv1_p', conv1_p, {'images'}, {'conv1_p'}, {'conv1f', 'conv1b'});
net.addLayer('relu1_p', dagnn.ReLU(), {'conv1_p'}, {'conv1x_p'});
norm1_p = dagnn.LRN('param', [5 1 0.0001/5 0.75]);
net.addLayer('norm1_p', norm1_p, {'conv1x_p'}, {'norm1_p'});
pool1_p = dagnn.Pooling('method', 'max', 'poolSize', [3 3],'pad', 0, 'stride', 2);
net.addLayer('pool1_p', pool1_p, {'norm1_p'}, {'pool1_p'});

conv2_p = dagnn.Conv('size', [5 5 48 256], 'pad', 2, 'stride', 1, 'hasBias', true);
net.addLayer('conv2_p', conv2_p, {'pool1_p'}, {'conv2_p'}, {'conv2f', 'conv2b'});
net.addLayer('relu2_p', dagnn.ReLU(), {'conv2_p'}, {'conv2x_p'});
norm2_p = dagnn.LRN('param', [5 1 0.0001/5 0.75]);
net.addLayer('norm2_p', norm2_p, {'conv2x_p'}, {'norm2_p'});
pool2_p = dagnn.Pooling('method', 'max', 'poolSize', [3 3], 'pad', 0, 'stride', 2);
net.addLayer('pool2_p', pool2_p, {'norm2_p'}, {'pool2_p'});

conv3_p = dagnn.Conv('size', [3 3 256 384], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv3_p', conv3_p, {'pool2_p'}, {'conv3_p'}, {'conv3f', 'conv3b'});
net.addLayer('relu3_p', dagnn.ReLU(), {'conv3_p'}, {'conv3x_p'});

conv4_p = dagnn.Conv('size', [3 3 192 384], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv4_p', conv4_p, {'conv3x_p'}, {'conv4_p'}, {'conv4f', 'conv4b'});
net.addLayer('relu4_p', dagnn.ReLU(), {'conv4_p'}, {'conv4x_p'});

conv5_p = dagnn.Conv('size', [3 3 192 256], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv5_p', conv5_p, {'conv4x_p'}, {'conv5_p'}, {'conv5f', 'conv5b'});
net.addLayer('relu5_p', dagnn.ReLU(), {'conv5_p'}, {'conv5x_p'});
pool5_p = dagnn.Pooling('method', 'max', 'poolSize', [3 3], 'pad', 0, 'stride', 2);
net.addLayer('pool5_p', pool5_p, {'conv5x_p'}, {'pool5_p'});

%% concat

net.addLayer('concat' , dagnn.Concat(), {'pool5', 'pool5_p'}, {'pool5_concat'});

%% fc

fc6_new = dagnn.Conv('size', [6 6 512 4096], 'pad', 0, 'stride', 1, 'hasBias', true);%need fix
net.addLayer('fc6_new', fc6_new, {'pool5_concat'}, {'fc6'}, {'filters6', 'biases6'});
net.addLayer('relu6', dagnn.ReLU(), {'fc6'}, {'fc6x'});
drop6 = dagnn.DropOut('rate', 0.5);
net.addLayer('drop6', drop6, {'fc6x'}, {'fc6x_dropout'});

fc7_new = dagnn.Conv('size', [1 1 4096 4096], 'pad', 0, 'stride', 1, 'hasBias', true);
net.addLayer('fc7_new', fc7_new, {'fc6x_dropout'}, {'fc7'}, {'filters7', 'biases7'});
net.addLayer('relu7', dagnn.ReLU(), {'fc7'}, {'fc7x'});
drop7 = dagnn.DropOut('rate', 0.5);
net.addLayer('drop7', drop7, {'fc7x'}, {'fc7x_dropout'});

fc7_newb = dagnn.Conv('size', [1 1 4096 4096], 'pad', 0, 'stride', 1, 'hasBias', true);
net.addLayer('fc7_newb', fc7_newb, {'fc7x_dropout'}, {'fc7b'}, {'filters7b', 'biases7b'});
net.addLayer('relu7b', dagnn.ReLU(), {'fc7b'}, {'fc7bx'});
drop7b = dagnn.DropOut('rate', 0.5);
net.addLayer('drop7b', drop7b, {'fc7bx'}, {'fc7bx_dropout'});

fc8_shapes = dagnn.Conv('size', [1 1 4096 4], 'pad', 0, 'stride', 1, 'hasBias', true);
net.addLayer('fc8_shapes', fc8_shapes, {'fc7bx_dropout'}, {'fc8'}, {'filters8', 'biases8'});

net.addLayer('lossl1', dagnn.LossL1(), {'fc8', 'bboxes_gt_scaled'}, 'objective');

%% params
net.initParams();
alex_net = load('../model/imagenet-caffe-alex.mat');
alex_net = dagnn.DagNN.fromSimpleNN(alex_net);
net.params(net.getParamIndex('conv1f')).value = alex_net.params(alex_net.getParamIndex('conv1f')).value;
net.params(net.getParamIndex('conv1b')).value = alex_net.params(alex_net.getParamIndex('conv1b')).value;

net.params(net.getParamIndex('conv2f')).value = alex_net.params(alex_net.getParamIndex('conv2f')).value;
net.params(net.getParamIndex('conv2b')).value = alex_net.params(alex_net.getParamIndex('conv2b')).value;

net.params(net.getParamIndex('conv3f')).value = alex_net.params(alex_net.getParamIndex('conv3f')).value;
net.params(net.getParamIndex('conv3b')).value = alex_net.params(alex_net.getParamIndex('conv3b')).value;

net.params(net.getParamIndex('conv4f')).value = alex_net.params(alex_net.getParamIndex('conv4f')).value;
net.params(net.getParamIndex('conv4b')).value = alex_net.params(alex_net.getParamIndex('conv4b')).value;

net.params(net.getParamIndex('conv5f')).value = alex_net.params(alex_net.getParamIndex('conv5f')).value;
net.params(net.getParamIndex('conv5b')).value = alex_net.params(alex_net.getParamIndex('conv5b')).value;
clear alex_net

frozenParamIdx = net.getParamIndex({'conv1f','conv1b',...
    'conv2f','conv2b','conv3f','conv3b',...
    'conv4f','conv4b','conv5f','conv5b'});
frozenRate = 0;
[net.params(frozenParamIdx).learningRate] = deal(frozenRate);
net.mode = 'normal';
