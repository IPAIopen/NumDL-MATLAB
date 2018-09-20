%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [Y,Yall,dA] = ResNetForward(Kb,Y,param)
%
% Forward propagation through ResNet
%
% Y{j+1} = P{j}*Y{j} + h*act(K{j}'*Y{j} + b{j})
%
% where P{j} are given and K,b are the weights to be learned
%
% Inputs:
%
%  Kb    - vector of weights, parsed by vec2cellResNet
%  Y     - input features
%  param - struct describing the networks. Required fields
%          h    - time step size
%          P    - cell, P = opEye for ReseNet, P = opZeros for NeuralNet
%          n    - 2xnt matrix, dimensions of K for each layer
%
% Outputs:
%
% Y      - output features
% Yall   - features at hidden layers
% dA     - cell, derivatives of activations at hidden layers
%                                     (needed for derivative computation)
function [Y,Yall,dA] = ResNetForward(Kb,Y,param)

[K,b] = vec2cellResNet(Kb,param.n);

h   = param.h;
P   = param.P;
N   = numel(param.P);
act = param.act;

% store intermediates
[Yall,dA] = deal(cell(N+1,1));
if nargout>1
    Yall{1} = Y;
end

% do the forward propagation
for j=1:N
    [Aj,dAj] = act(K{j}*Y + b{j});
    Y        = P{j}*Y + h*Aj;
    
    if nargout>1; Yall{j+1} = Y; end
    if nargout>2; dA{j}= dAj;    end    
end