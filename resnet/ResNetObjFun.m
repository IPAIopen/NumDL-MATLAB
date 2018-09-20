%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [Ec,dE,H] = ResNetObjFun(x,Y,C,m)
%
% evaluates resnet and computes cross entropy, gradient and approx. Hessian
%
% Let x = [K(:);b(:);W(:)], we compute
%
% E(x) = E(W*Z,C),   where Z = ResNetForward(Kb,Y0)
%
% Inputs:
%
%   x          - current iterate, x=[K(:);b(:);W(:)]
%   Y          - input features
%   C          - class probabilities
%   nKb        - number of network parameters
%   paramResnet- param type for ResNet
%   paramReg   - struct, parameter describing regularizer
%
% Output:
%
%   Ec - current value of loss function
%   dE - gradient w.r.t. K,b,W, vector
%   H  - approximate Hessian, H=J'*d2ES*J, function_handle
function [Ec,dE,H] = ResNetObjFun(x,Y,C,nKb,paramResnet,paramReg)


if nargin==0
    exResNet_Peaks
    return;
end

[nf,nex] = size(Y);
nc     = size(C,1);

% split x into K,b,W
x = x(:);
Kb = x(1:nKb);
W = reshape(x(nKb+1:end),[],nc);

% evaluate layer
[Z,Yall,dA]  = ResNetForward(Kb,Y,paramResnet);

% call cross entropy
[Ec,dEW,d2EW,dEZ,d2EZ] = softMax(W,Z,C);

% add regularizer
if not(exist('paramReg','var')) || not(isstruct(paramReg))
    dS = 0; d2S = @(x) 0;
else
    [Sc,dS,d2S] = genTikhonov(x,paramReg);
    Ec = Ec + Sc;
end

if nargout>1
    [dEK,dEb] = dResNetMatVecT(reshape(dEZ,[],nex),Kb,Yall,dA,paramResnet);
    dE  = [cell2vec(dEK); cell2vec(dEb(:)); dEW(:)] + dS;
end

H = [];
if nargout>2
    H = @(x) HessMat(x,nKb,Yall,dA,Y,d2EW,d2EZ,Kb,d2S,paramResnet);
end

function Hx = HessMat(x,nKb,Yall,dA,Y,d2EW,d2EZ,Kb,d2S,param)
x   = x(:);
dKb = x(1:nKb);
dW  = x(nKb+1:end);


% compute Jac*x
JKbx = dResNetMatVec(dKb,0*Y,Kb,Yall,dA,param);
tt   = d2EZ(JKbx);
[J1,J2] = dResNetMatVecT(reshape(tt,size(JKbx)),Kb,Yall,dA,param);
Hx1  = [cell2vec(J1);cell2vec(J2)];

Hx2  = d2EW(dW);

% stack result
Hx = [Hx1(:); Hx2(:)] + d2S(x);








