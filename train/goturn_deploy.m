function net = goturn_deploy(net)
%GOTURN_DEPLOY  Deploy a CNN

isDag = isa(net, 'dagnn.DagNN') ;
if isDag
    dagRemoveLayersOfType(net, 'dagnn.Loss') ;
    dagRemoveLayersOfType(net, 'dagnn.DropOut') ;
else
    net = simpleRemoveLayersOfType(net, 'softmaxloss') ;
    net = simpleRemoveLayersOfType(net, 'dropout') ;
end

end



% -------------------------------------------------------------------------
function layers = dagFindLayersOfType(net, type)
% -------------------------------------------------------------------------
layers = [] ;
for l = 1:numel(net.layers)
    if isa(net.layers(l).block, type)
        layers{1,end+1} = net.layers(l).name ;
    end
end
end

% -------------------------------------------------------------------------
function dagRemoveLayersOfType(net, type)
% -------------------------------------------------------------------------
names = dagFindLayersOfType(net, type) ;
for i = 1:numel(names)
    layer = net.layers(net.getLayerIndex(names{i})) ;
    net.removeLayer(names{i}) ;
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
end
end
