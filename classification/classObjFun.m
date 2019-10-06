%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [J,dJ,H] =  classObjFun(W,Z,C,paramRegW,varargin)
%
% classification objective function
%
%  J(W) = softMax(W,Z,C) + genTikhonow(W,paramRegW)
%
% Input:
%  W   - current weights
%  Z   - feature matrix
%  C   - labels
%  paramRegW  - struct, parameters for regularizer
%
% Output:
%  J   - current value of objective fuction
%  dJ  - gradient
%  H   - approximate Hessian
%
function [J,dJ,H] =  classObjFun(W,Z,C,paramRegW,varargin)
loss = @softMax;
for k=1:2:length(varargin)    % overwrites default parameter
  eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end;


[J,dJ,H] = loss(W,Z,C);

if exist('paramRegW','var') && isstruct(paramRegW)
    [Sc,dS,d2S] = genTikhonov(W,paramRegW);
    J = J+ Sc;
    dJ = dJ + dS;
    H = @(x) H(x) + d2S(x);
end


