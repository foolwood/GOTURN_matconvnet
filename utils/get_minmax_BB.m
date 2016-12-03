% -------------------------------------------------------------------------------------------------
function [bb] = get_minmax_BB(region)
%get_minmax_BB computes minmax bbox from the rotated one (REGION)
% -------------------------------------------------------------------------------------------------
if isa(region,'cell')
    bb = cellfun(@get_axis_aligned_BB,region,'UniformOutput',false);
else
    if numel(region) < 8,bb = [];return;end
    bb = [min(region(:,1:2:end),[],2),...
        min(region(:,2:2:end),[],2),...
        max(region(:,1:2:end),[],2),...
        max(region(:,2:2:end),[],2)]-1;
end
end
