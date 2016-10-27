classdef LossL1 < dagnn.Loss
%LossL1  L1 loss
%  `LossL1.forward({x, x0})` computes the  L1 distance 
%  between `x` and `x0`.
%
%  Here the smooth L1 loss between two vectors is defined as:
%
%     Loss = sum_i f(x_i - x0_i).
%
%  where f is the function (following the Faster R-CNN definition):
%
%   f(delta) = |delta|
%             
%  In practice, `x` and `x0` can pack multiple instances as 1 x 1 x C
%  x N arrays (or simply C x N arrays).

  methods
    function outputs = forward(obj, inputs, params)
      
      delta = inputs{1} - inputs{2} ;
      absDelta = abs(delta) ;

      % Mutliply by instance weights and sum.
      outputs{1} = sum(absDelta(:)) ;

      % Accumulate loss statistics.
      n = obj.numAveraged ;
      m = n + 1 + 1e-9 ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
    % Function derivative:
    %        
    %  f'(x) = sign(x),                 otherwise.
    %          

      delta = sign(inputs{1} - inputs{2}) ;

      derInputs = {delta .* derOutputs{1}, [], []} ;
      derParams = {} ;
    end

    function obj = LossL1(varargin)
      obj.load(varargin) ;
      obj.loss = 'l1';
    end
  end
end
