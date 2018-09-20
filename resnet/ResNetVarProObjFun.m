%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [Ec,dE,H] = ResNetVarProObjFun(x,Y,C,m)
%
% evaluates resnet, solves classification problem, and computes cross entropy, 
% gradient and approx. Hessian
%
% Let x = [K(:);b(:)], we compute
%
% E(x) = E(W(x)*Z,C),   where Z = ResNetForward(Y0,Kb)
%
% Inputs:
%
%   x - current iterate, x=[K(:);b(:);W(:)]
%   Y - input features
%   C     - class probabilities
%   nKb   - number of network parameters
%   paramCl     - struct, paramters for newtoncg used to classify
%   paramResnet - struct, parameter describing resnet
%   paramRegKb  - struct, parameter describing regularizer for K and b
%   paramRegW   - struct, parameter describing regularizer for W
%
% Output:
%
%   Ec - current value of loss function
%   dE - gradient w.r.t. K,b,W, vector
%   H  - approximate Hessian, H=J'*d2ES*J, function_handle
function [Ec,dE,H] = ResNetVarProObjFun(x,Y,C,nKb,paramCl,paramResnet,paramRegKb,paramRegW)
if nargin==0
    exResNet_PeaksVarPro
    return;
end

[nf,nex] = size(Y);
nc     = size(C,1);

% split x into K,b,W
x = x(:);
Kb = x(1:nKb);

% evaluate Resnet
[Z,Yall,dA]  = ResNetForward(Kb,Y,paramResnet);

% solve classification problem
if exist('paramRegW','var')
    fctn = @(W,varargin) classObjFun(W,Z,C,paramRegW);
else
    fctn = @(W,varargin) softMax(W,Z,C);
end
WOpt = newtoncg(fctn,zeros(nc*(size(Z,1)+1),1),paramCl);


% call cross entropy
[Ec,dEW,d2EW,dEZ,d2EZ] = softMax(WOpt,Z,C);

% add regularizer
if not(exist('paramRegKb','var')) || not(isstruct(paramRegKb))
    dS = 0; d2S = @(x) 0;
else
    [Sc,dS,d2S] = genTikhonov(x,paramRegKb);
    Ec = Ec + Sc;
end
if nargout>1
    [dEK,dEb] = dResNetMatVecT(reshape(dEZ,[],nex),Kb,Yall,dA,paramResnet);
    dE  = [cell2vec(dEK); cell2vec(dEb(:))] + dS;
end

H = [];
if nargout>2
    H = @(x) HessMat(x,nKb,Yall,dA,Y,d2EZ,Kb,d2S,paramResnet);
end

function Hx = HessMat(x,nKb,Yall,dA,Y,d2EZ,Kb,d2S,param)
x   = x(:);
dKb = x(1:nKb);


% compute Jac*x
JKbx = dResNetMatVec(dKb,0*Y,Kb,Yall,dA,param);
tt   = d2EZ(JKbx);
[J1,J2] = dResNetMatVecT(reshape(tt,size(JKbx)),Kb,Yall,dA,param);
Hx1  = [cell2vec(J1);cell2vec(J2)];


% stack result
Hx = Hx1(:) + d2S(x);








