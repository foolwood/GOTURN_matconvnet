function r_tracker(img_files, ground_truth, show_visualization)
close all
if show_visualization,  %create video interface
    update_visualization = show_video(img_files);
end

% image_bank = vl_imreadjpeg(img_files);
% image_prev = image_bank{1};

for frame = 2:numel(img_files),
  
%     image_curr = image_bank{frame};
    
    if show_visualization,
        stop = update_visualization(frame, ground_truth(frame,:),[]);
        if stop, break, end
    end
    
end

end %%function
