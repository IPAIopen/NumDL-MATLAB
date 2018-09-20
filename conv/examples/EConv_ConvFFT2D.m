%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% demo for 2D convolution
close all; clear all;

[Y,C] = setupMNIST(1);
Y = Y - mean(Y(:));
nImg  = [28 28]; % size of image
sTheta = [5 5];  % size of convolution stencil
theta = randn(sTheta);
theta = theta - mean(theta(:));

%% 
K = conv2D(nImg,sTheta,theta,Y);
Z = K(Y);

%%
fig = figure(1); clf;
fig.Name = sprintf('%s',mfilename);
subplot(1,3,1);
imagesc(reshape(Y,nImg));
axis square off;
colormap(flipud(colormap('gray')))
title('input image');


subplot(1,3,2);
imagesc(theta);
axis square off;
colormap('gray')
title('convolution kernel');


subplot(1,3,3);
imagesc(reshape(Z,nImg));
axis square off;
colormap(flipud(colormap('gray')))
title('output image');
