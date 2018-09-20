%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
%   demo for instancenorm
%
close all; clear all;

nex = 8;
[Y,C] = setupMNIST(nex,1);

%%
param.dir = 1;
param.epsilon = 1e-5;
Yn = normLayer(Y,param);

%%
fig = figure; clf;
fig.Name = [mfilename ': batch'];
subplot(2,1,1)
montageArray(reshape(Y,28,28,[]),nex);
axis equal tight
colorbar

subplot(2,1,2)
montageArray(reshape(Yn,28,28,[]),nex);
axis equal tight
colorbar


%%
fig = figure; clf;
fig.Name = [mfilename ': first image'];
subplot(1,2,1)
montageArray(reshape(Y(:,1),28,28,[]));
axis equal tight
colorbar
set(gca,'FontSize',20)
title('original');

subplot(1,2,2)
montageArray(reshape(Yn(:,1),28,28,[]));
axis equal tight
title('after instance norm')
set(gca,'FontSize',20)
colorbar