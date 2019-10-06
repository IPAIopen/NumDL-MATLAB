%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% logistic regression for the circle example. Here we see that a nonlinear
% transformation of the feature space is required. The problem can also be
% used to illustrate feature engineering.
%
close all; clear all; clc;

%% get peaks data

[Y, C] = setupEllipses(1000);

% % normalize
% Y = Y - mean(Y,2);
% Y = Y./std(Y,[],2);

% % add features
% r    = sqrt(sum(Y.^2,1)); 
% alph = acos(Y(1,:)./r);
% Y = [r;alph];

% show data
figure(1); clf;
col1 = [0 0.44 0.75];
col2 = [0.85 0.32 0.1];
p1 = plot(Y(1,C==1),Y(2,C==1),'.','MarkerSize',10);
p1.Color=col1;
hold on
p2 = plot(Y(1,C==0),Y(2,C==0),'.','MarkerSize',10);
p2.Color=col2;
axis equal tight


%% split into training and validation


numTrain = size(Y, 2)*0.80;
idx = randperm(size(Y,2));
idxTrain = idx(1:numTrain);
idxValid = idx(numTrain+1:end);

YTrain = Y(:,idxTrain);
CTrain = C(:,idxTrain);

YValid = Y(:,idxValid);
CValid = C(:,idxValid);

nf = size(Y,1);
nc = size(C,1);
%% optimize
% m = 640/20;
W0   = randn(nc,nf+1);

paramRegW = struct('L',speye(numel(W0)),'lambda',1e-3);
fctn = @(x,varargin) classObjFun(x,YTrain,CTrain,paramRegW,'loss',@logRegression);
param = struct('maxIter',30,'maxStep',1,'tolCG',1e-3,'maxIterCG',100);
WOpt = newtoncg(fctn,W0(:),param);
%%
WOpt = reshape(WOpt,nc,[]);
Strain = WOpt*padarray(YTrain,[1,0],1,'post');
Svalid = WOpt*padarray(YValid,[1,0],1,'post');
htrain = Strain>0;
hvalid = Svalid>0;

trainErr = 100*nnz(abs(CTrain-htrain))/2/nnz(CTrain);
valErr   = 100*nnz(abs(CValid-hvalid))/2/nnz(CValid);
%%
x = linspace(-2.5,2.5,201);
[Xg,Yg] = ndgrid(x);
S = WOpt * padarray([vec(Xg)'; vec(Yg)'],[1,0],1,'post');
posInd = (S>0);
negInd = (0 >= S);
P = 0*S;
P(posInd) = 1./(1+exp(-S(posInd)));
P(negInd) = exp(S(negInd))./(1+exp(S(negInd)));
Cpred = P > 0.5;
img = reshape(Cpred,size(Xg));
%%
col1 = [0 0.44 0.75];
col2 = [0.85 0.32 0.1];
figure(2);clf;
ih = imagesc(x,x,img')
ih.AlphaData = .5
colormap([col1;col2]);
hold on;
p1 = plot(YTrain(1,CTrain==1),YTrain(2,CTrain==1),'.','MarkerSize',10);
p1.Color=col1;
p2 = plot(YTrain(1,CTrain==0),YTrain(2,CTrain==0),'.','MarkerSize',10);
p2.Color=col2;
title(sprintf('train  error %1.2f%% val error %1.2f%%',trainErr,valErr));



