%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [K,b] = vec2cellResNet(v,n)
%
% splits vector v into cell arrays associated with K{j} and b{j} in
% ResNetForward.
%
% Inputs:
%  
%  v  - weight vector
%  n  - descriptions of kernel size in each layer
%
% Outputs:
%
%  K,b - cell arrays of weights.

function [K,b] = vec2cellResNet(v,n)

nt = size(n,2);
cnt = 0;

K = cell(nt,1);
b = cell(nt,1);

% first get the Ks
for k=1:nt
   nk = prod(n(:,k));
   K{k} = reshape( v(cnt+(1:nk)), n(2,k),n(1,k));
   cnt = cnt + nk;
end
    
% now get the bs
for k=1:nt
   nb = n(end,k);
   b{k} = v(cnt+(1:nb));
   cnt = cnt + nb;
end

   

