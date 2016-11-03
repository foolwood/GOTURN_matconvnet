function update_visualization_func = show_video(img_files)
%SHOW_VIDEO
%   Visualizes a tracker in an interactive figure, given a cell array of
%   image file names, their path, and whether to resize the images to
%   half size or not.
%
%   This function returns an UPDATE_VISUALIZATION function handle, that
%   can be called with a frame number and a bounding box [x, y, width,
%   height], as soon as the results for a new frame have been calculated.
%   This way, your results are shown in real-time, but they are also
%   remembered so you can navigate and inspect the video afterwards.
%   Press 'Esc' to send a stop signal (returned by UPDATE_VISUALIZATION).
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/


	%store one instance per frame
	num_frames = numel(img_files);
	boxes1 = cell(num_frames,1);
    boxes2 = cell(num_frames,1);
    boxes3 = cell(num_frames,1);

	%create window
	[fig_h, axes_h, unused, scroll] = videofig(num_frames, @redraw, [], [], @on_key_press);  %#ok, unused outputs
	set(fig_h, 'Name', 'Tracker - GOTURN')
	axis off;
	
	%image and rectangle handles start empty, they are initialized later
	im_h = [];
	rect1_h = [];
    rect2_h = [];
    rect3_h = cell(10,1);
    hsv_color = hsv;
    hsv_color = hsv_color(randperm(64,10),:);
	
	update_visualization_func = @update_visualization;
	stop_tracker = false;
	

	function stop = update_visualization(frame, box1, box2,box3cell)
		%store the tracker instance for one frame, and show it. returns
		%true if processing should stop (user pressed 'Esc').
		boxes1{frame} = box1;
        boxes2{frame} = box2;
        boxes3{frame} = box3cell;
		scroll(frame);
		stop = stop_tracker;
	end

	function redraw(frame)
		%render main image
		im = imread(img_files{frame});
		
		if isempty(im_h),  %create image
			im_h = imshow(im, 'Border','tight', 'InitialMag',200, 'Parent',axes_h);
		else  %just update it
			set(im_h, 'CData', im)
		end
		hold on
		%render target bounding box for this frame
		if isempty(rect1_h),  %create it for the first time
			rect1_h = plot([0,0,0,0,0], [0,0,0,0,0], 'g','LineWidth',2, 'Parent',axes_h);
		end
        if ~isempty(boxes1{frame}),
            set(rect1_h, 'Visible', 'on', 'XData', boxes1{frame}([1,3,5,7,1]), 'YData', boxes1{frame}([2,4,6,8,2]));
        else
            set(rect1_h, 'Visible', 'off');
        end
        


        
        
        for i = 1:10
            if isempty(rect3_h{i}),  %create it for the first time
                rect3_h{i} = rectangle('Position',[0,0,1,1], 'EdgeColor',hsv_color(i,:), 'Parent',axes_h);
            end
            if numel(boxes3{frame})>=i && ~isempty(boxes3{frame}),
                set(rect3_h{i}, 'Visible', 'on', 'Position', boxes3{frame}{i});
            else
                set(rect3_h{i}, 'Visible', 'off');
            end
        end
        %render target bounding box for this frame
        if isempty(rect2_h),  %create it for the first time
            rect2_h = rectangle('Position',[0,0,1,1], 'EdgeColor','y', 'LineWidth',2,'Parent',axes_h);
        end
        if ~isempty(boxes2{frame}),
            set(rect2_h, 'Visible', 'on', 'Position', boxes2{frame});
        else
            set(rect2_h, 'Visible', 'off');
        end
        
	end

	function on_key_press(key)
		if strcmp(key, 'escape'),  %stop on 'Esc'
			stop_tracker = true;
		end
	end

end

