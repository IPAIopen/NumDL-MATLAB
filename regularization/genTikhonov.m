%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
%[R,dR,d2R] = genTikhonov(W,param)
%
% R(W) = 0.5*lambda*|L*W|^2
%
% where size(W)=size(L,2)*nc. 
%
% Input:
%   W     - weights, will be reshaped internally
%   param - struct with additional parameters. Required fields:
%           L      - regularization operator 
%           lambda - regularization parameter
%  
% Output:
%   Rc    - value of regularizer
%   dR    - gradient
%   d2R   - Hessian, as function handle

function[R,dR,d2R] = genTikhonov(W,param)

if nargin==0
    help(mfilename);
    return
end
if not(isfield(param,'lambda')); param.lambda = 1; end

L       = param.L;
lambda  = param.lambda;
W       = reshape(W,size(L,2),[]);

LW = L*W;
R  = 0.5* lambda * (LW(:)'*LW(:));
dR = lambda * L'*LW;
dR = dR(:);

mat    = @(X) reshape(X,size(W));
vec    = @(X) X(:);
d2Rmat = lambda*(L'*L);
d2R    = @(X) vec(d2Rmat*mat(X));