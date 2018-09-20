%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% Multinomial regression for the peaks example. Here we see that a nonlinear
% transformation of the feature space is required.
%
close all; clear all; clc;

%% get peaks data
np = 8000;  % num of points sampled
nc = 5;     % num of classes
ns = 256;   % length of grid

[Y, C] = setupPeaks(np, nc, ns);

numTrain = size(Y, 2)*0.80;
idx = randperm(numTrain);
idxTrain = idx(1:numTrain);
idxValid = idx(numTrain+1:end);

YTrain = Y(:,idxTrain);
CTrain = C(:,idxTrain);

YValid = Y(:,idxValid);
CValid = C(:,idxValid);

[YTest, CTest] = setupPeaks(2000, nc, ns);

nf = size(Y,1);
nc = size(C,1);
%% optimize
% m = 640/20;
W0   = randn(nc,3);

paramRegW = struct('L',speye(numel(W0)),'lambda',1e-3);
fctn = @(x,varargin) classObjFun(x,YTrain,CTrain,paramRegW);
param = struct('maxIter',30,'maxStep',1,'tolCG',1e-3,'maxIterCG',100);
WOpt = newtoncg(fctn,W0(:),param);
%%
WOpt = reshape(WOpt,nc,[]);
Strain = WOpt*padarray(YTrain,[1,0],1,'post');
S      = WOpt*padarray(YTest,[1,0],1,'post');
htrain = exp(Strain)./sum(exp(Strain),1);
h      = exp(S)./sum(exp(S),1);

% Find the largesr entry at each row
[~,ind] = max(h,[],1);
Cv = zeros(size(CTest));
Ind = sub2ind(size(Cv),ind,1:size(Cv,2));
Cv(Ind) = 1;
[~,ind] = max(htrain,[],1);
Cpred = zeros(size(CTrain));
Ind = sub2ind(size(Cpred),ind,1:size(Cpred,2));
Cpred(Ind) = 1;
5
trainErr = 100*nnz(abs(CTrain-Cpred))/2/nnz(CTrain);
valErr   = 100*nnz(abs(Cv-CTest))/2/nnz(Cv);
%%
x = linspace(-3,3,201);
[Xg,Yg] = ndgrid(x);
Z = WOpt * padarray([vec(Xg)'; vec(Yg)'],[1,0],1,'post');
h      = exp(Z)./sum(exp(Z),1);

[~,ind] = max(h,[],1);
Cpred = zeros(5,numel(Xg));
Ind = sub2ind(size(Cpred),ind,1:size(Cpred,2));
Cpred(Ind) = 1;
img = reshape((1:5)*Cpred,size(Xg));
%%
figure(1);
imagesc(x,x,img')
title(sprintf('train %1.2f%% val %1.2f%%',trainErr,valErr));



