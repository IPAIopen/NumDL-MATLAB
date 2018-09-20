%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% Training a single layer neural network for Peaks example
%
% Here we optimize the coupled objective function first with steepest
% descent and then with newtoncg. Compare this example with
% ESingleLayer_PeaksVarPro
%
clear;
%% get peaks data
np = 4000;  % num of points sampled
nc = 5;     % num of classes
ns = 256;   % length of grid

[Y, C] = setupPeaks(np, nc, ns);
[Yv, Cv] = setupPeaks(2000, nc, ns);

nf = size(Y,1);
nc = size(C,1);
%% optimize
m = 40;
K0 = randn(m,nf);
b0 = randn(1);
W0 = randn(nc,m+1);


%% optimize
paramReg = struct('L',1,'lambda',1e-4,'nc',1);
paramSL = struct('act',@sinActivation);
fctn = @(x,varargin) singleLayerNNObjFun(x,Y,C,m,paramSL,paramReg);
x0   = [K0(:); b0(:); W0(:)];

param.lr = 1e-1*ones(50,1);
    param.n  = size(Y,2);
    param.batchSize = 4;
    param.momentum=0.0;
    
xOpt = sgd(fctn,x0,param);
%%
%%

KOpt = reshape(xOpt(1:m*nf),m,nf);
bOpt = xOpt(m*nf+1);
WOpt = reshape(xOpt(m*nf+2:end),nc,[]);
%%

St = WOpt*padarray(singleLayer(KOpt,bOpt,Y,paramSL),[1 0],1,'post');
Sv = WOpt*padarray(singleLayer(KOpt,bOpt,Yv,paramSL),[1 0],1,'post');
htrain = exp(St)./sum(exp(St),1);
h      = exp(Sv)./sum(exp(Sv),1);

% Find the largesr entry at each row
[~,ind] = max(h,[],1);
Cvpred = zeros(size(Cv));
Ind = sub2ind(size(Cvpred),ind,1:size(Cvpred,2));
Cvpred(Ind) = 1;
[~,ind] = max(htrain,[],1);
Cpred = zeros(size(C));
Ind = sub2ind(size(Cpred),ind,1:size(Cpred,2));
Cpred(Ind) = 1;

trainErr = 100*nnz(abs(C-Cpred))/2/nnz(C);
valErr   = 100*nnz(abs(Cv-Cvpred))/2/nnz(Cv);

%%
x = linspace(-3,3,201);
[Xg,Yg] = ndgrid(x);
Z = WOpt*[singleLayer(KOpt,bOpt,[Xg(:)';Yg(:)'],paramSL); ones(1,numel(Xg))];
h      = exp(Z)./sum(exp(Z),1);

[~,ind] = max(h,[],1);
Cpred = zeros(5,numel(Xg));
Ind = sub2ind(size(Cpred),ind,1:size(Cpred,2));
Cpred(Ind) = 1;
img = reshape((1:5)*Cpred,size(Xg));
%%
figure(2);clf
imagesc(x,x,img')
title(sprintf('training error %1.2f%% validation error %1.2f%%',trainErr,valErr));


