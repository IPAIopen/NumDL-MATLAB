%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
%   demo for batchnorm
%
close all; clear all;

nex = 8;
[Y,C] = setupMNIST(nex,1);
Y = reshape(Y,28,28,[]);
%%
param.dir = 3;
param.epsilon = 1e-5;
Yn = normLayer(Y,param);

%%
figure(1); clf;
subplot(2,1,1)
montageArray(Y,nex);
axis equal tight
colorbar

subplot(2,1,2)
montageArray(Yn,nex);
axis equal tight
colorbar


%%
figure(2); clf;
subplot(1,2,1)
montageArray(reshape(Y(:,:,1),28,28,[]));
axis equal tight
colorbar
set(gca,'FontSize',20)
title('original');

subplot(1,2,2)
montageArray(reshape(Yn(:,:,1),28,28,[]));
axis equal tight
title('after batch norm')
set(gca,'FontSize',20)
colorbar
%  caxis([-. .1])close all; clear all;

%%
[Y,C] = setupMNIST(16);
Y = reshape(Y,28,28,[]);

%%
param.dir = 3;
param.epsilon = 1e-5;
Yn = normLayer(Y,param);

%%
figure(1); clf;
subplot(2,1,1)
montageArray(Y,16);
axis equal tight
colorbar

subplot(2,1,2)
montageArray(Yn,16);
axis equal tight
colorbar


%%
figure(2); clf;
subplot(1,2,1)
montageArray(Y(:,:,1))
axis equal tight
colorbar
set(gca,'FontSize',20)
title('original');

subplot(1,2,2)
montageArray(Yn(:,:,1))
axis equal tight
title('after batch norm')
set(gca,'FontSize',20)
colorbar
%  caxis([-. .1])