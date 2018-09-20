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
%  dKb   - vector, perturbation of weights
%  dY    - matrix, perturbation of input features
%  Kb    - vector, current weights
%  Yall  - cell, hidden features
%  dA    - cell, derivative of activations at all layers
%  param - struct, description of ResNet
% 
% Outputs: 
%  
%  dY    - JKb*dKb + JY*dY
%
% for forward propagation, see ResNetForward.m

function dY = dResNetMatVec(dKb,dY,Kb,Yall,dA,param)



[K,~]   = vec2cellResNet(Kb,param.n);
[dK,db] = vec2cellResNet(dKb,param.n);

h = param.h;
P = param.P;
N = numel(param.P);

for j=1:N
    dY     = P{j}*dY + h*dA{j}.*(dK{j}*Yall{j} + K{j}*dY +db{j});
end