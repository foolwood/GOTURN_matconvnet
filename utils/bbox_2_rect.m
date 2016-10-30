function rect = bbox_2_rect(bbox)%%double format output
rect = double([bbox(:,1)+1,bbox(:,2)+1,bbox(:,3)-bbox(:,1),bbox(:,4)-bbox(:,2)]);
end %%function