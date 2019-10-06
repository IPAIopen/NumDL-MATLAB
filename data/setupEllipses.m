%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [Y,C] = setupEllipses(np, nc, ns)
%
% setup data for ellipses example
%
function[Y,C] = setupEllipses(np)
% generates Ellipses example
if not(exist('np','var')) || isempty(np)
    np = 1000;
end

% get training data
rb     = rand(np,1);
thetab = rand(np,1)*2*pi; 
xb1 = rb .* cos(thetab);
xb2 = .4*rb .* sin(thetab);
rr     = 1+rand(np,1);
thetar = rand(np,1)*2*pi; 
xr1 = rr .* cos(thetar);
xr2 = .4*rr .* sin(thetar);
Y = [[xb1; xr1],[xb2; xr2]]';
C  =  [ones(1,np), zeros(1,np)];

