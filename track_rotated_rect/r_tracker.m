function r_tracker(img_files, ground_truth, show_visualization)
close all
if show_visualization,  %create video interface
    update_visualization = show_video(img_files);
end

image_bank = vl_imreadjpeg(img_files);
image_prev = image_bank{1};

for frame = 2:numel(image_bank),
  
    image_curr = image_bank{frame};
    
    if show_visualization,
        bbox_estimate_uncentered_cell = [];
        stop = update_visualization(frame, ground_truth(frame,:),[],bbox_estimate_uncentered_cell);
        if stop, break, end
        drawnow
    end
    
end

end %%function
