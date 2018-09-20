%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [A,dA] = smoothReluActivation(Y,varargin)
%
% smoothed relu activation function A = smoothReluActivation(Y). The idea
% is to use a quadratic model close to the origin to ensure
% differentiability:
%
%            | y, if y>eta  
% sigma(y) = | 0, if y<eta
%            | a*y^2 + b*y + c, if -eta <= y <= eta
%
% where the coefficients a,b,c are chosen to ensure that sigma is
% continuously differentiable. 
%
% Input:
%  
%   Y - array of features
%
% Optional Input:
%
%   doDerivative - flag for computing derivative, set via varargin
%                  Ex: smoothReluActivation(Y,'doDerivative',0,'eta',.1);
%
% Output:
%
%  A  - activation
%  dA - derivatives

function [A,dA] = smoothRelU(Y,varargin)


if nargin==0
    runMinimalExample;
    return
end
eta          = 0.1;
doDerivative = nargout==2;
for k=1:2:length(varargin)    % overwrites default parameter
  eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

a = 1/(4*eta);
b = 0.5;
c = 0.25*eta;

dA = [];

A              = max(Y,0);
ii             = abs(Y)<=eta;
A(ii) = a.*Y(ii).^2 + b.*Y(ii) + c;

if doDerivative
    dA              = sign(A);
    dA(ii)          = 2*a*Y(ii) + b;
end



function runMinimalExample
Y  = linspace(-1,1,101);
[A,dA] = feval(mfilename,Y);

fig = figure(100);clf;
fig.Name = mfilename;
plot(Y,A,'linewidth',3);
hold on;
plot(Y,dA,'linewidth',3);
xlabel('y')
legend('smoothRelu(y)','smoothRelu''(y)')
axis equal