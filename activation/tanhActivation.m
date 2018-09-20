%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [A,dA] = tanhActivation(Y,varargin)
%
% hyperbolic tan activation function A = tanh(Y)
%
% Input:
%  
%   Y - array of features
%
% Optional Input:
%
%   doDerivative - flag for computing derivative, set via varargin
%                  Ex: tanhActivation(Y,'doDerivative',0);
%
% Output:
%
%  A  - activation
%  dA - derivatives
function [A,dA] = tanhActivation(Y,varargin)

if nargin==0
    runMinimalExample;
    return
end

doDerivative = nargout==2;
for k=1:2:length(varargin)    % overwrites default parameter
  eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end


dA = [];

A = tanh(Y);

if doDerivative
     dA = 1-A.^2;
end



function runMinimalExample
Y  = linspace(-3,3,101);
[A,dA] = feval(mfilename,Y);

fig = figure(100);clf;
fig.Name = mfilename;
plot(Y,A,'linewidth',3);
hold on;
plot(Y,dA,'linewidth',3);
xlabel('y')
legend('tanh(y)','1-tanh(y)^2')