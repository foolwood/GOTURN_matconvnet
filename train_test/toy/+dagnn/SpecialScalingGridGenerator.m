%DAGNN.SpecialScalingGridGenerator  Generate an anisotropic scaling + translation
%   grid for bilinear resampling.
%   This layer maps 1 x 1 x 4 x N transformation parameters corresponding to
%   the scale and translation y and x respectively, to 2 x Ho x Wo x N
%   sampling grids compatible with dagnn.BlilinearSampler.

%   2016 Qiang Wang
classdef SpecialScalingGridGenerator < dagnn.Layer

 properties
     Ho = 0;
     Wo = 0;
 end

  properties (Transient)
    % the grid --> this is cached
    % has the size: [2 x HoWo]
    xxyy ;
  end

  methods

    function outputs = forward(obj, inputs, ~)
      % input is a 1x1x4xN TENSOR corresponding to:
      % [  s1 0 ty ]
      % [  0 s2 tx ]
      %
      % OUTPUT is a 2xHoxWoxN grid

      % reshape the tfm params into matrices:
      T = inputs{1};
      % check shape:
      sz_T = size(T);
      assert(all(sz_T(1:3) == [1 1 4]), 'transforms have incorrect shape');
      nbatch = size(T,4);
      S = reshape(T(1,1,1:2,:), 2,1,nbatch); % x,y scaling
      t = reshape(T(1,1,3:4,:), 2,1,nbatch); % translation
      % generate the grid coordinates:
      useGPU = isa(T, 'gpuArray');
      if isempty(obj.xxyy)
        obj.initGrid(useGPU);
      end
      % transform the grid:
      g = bsxfun(@times, obj.xxyy, S); % scale
      g = bsxfun(@plus, g, t); % translate
      g = reshape(g, 2,obj.Ho,obj.Wo,nbatch);
      outputs = {g};
    end
    
    function outputSizes = getOutputSizes(obj, inputSizes)
      nBatch = inputSizes{1}(4);
      outputSizes = {[2, obj.Ho, obj.Wo, nBatch]};
    end

    function obj = SpecialScalingGridGenerator(varargin)
      obj.load(varargin);
      % get the output sizes:
      obj.Ho = obj.Ho;
      obj.Wo = obj.Wo;
      obj.xxyy = [];
    end

    function obj = reset(obj)
      reset@dagnn.Layer(obj) ;
      obj.xxyy = [] ;
    end

    function initGrid(obj, useGPU)
      % initialize the grid:
      % this is a constant
      xi = linspace(-1, 1, obj.Ho);
      yi = linspace(-1, 1, obj.Wo);
      [yy,xx] = meshgrid(yi,xi);
      xxyy_ = [xx(:), yy(:)]' ; % 2xM
      if useGPU
        obj.xxyy = gpuArray(xxyy_);
      end
      obj.xxyy = xxyy_ ;
    end

  end
end
