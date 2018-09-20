%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% dY = dResNetMatVec(dKb,dY,Kb,Yall,dA,param)
% 
% computes matrix vector product with Jacobian of ResNet
%
% Inputs:
% 
%  dY    - perturbation of output features
%  Kb    - current weights
%  Yall  - hidden features
%  dA    - derivative of activations at all layers
%  param - struct, description of ResNet
% 
% Outputs: 
%
%  dK    - cell, derivatives w.r.t. K
%  db    - cell, derivatives w.r.t. b
%  dY    - matrix, derivatives w.r.t. input features
%
% for forward propagation, see ResNetForward.m

function[dK,db,dY] = dResNetMatVecT(dY,Kb,Yall,dA,param)



[K,~]   = vec2cellResNet(Kb,param.n);
h       = param.h;
P       = param.P;
N       = numel(param.P);
[dK,db] = deal(cell(N,1));

for j=N:-1:1
    % get derivatives w.r.t. Kj and bj
    dK{j} = h*(dA{j}.*dY) * Yall{j}';
    db{j} = sum(h*(dA{j}.*dY),2);
    
    % integrate backwards in time
    dY    = P{j}'*dY + h*K{j}'*(dA{j}.*dY);
end



