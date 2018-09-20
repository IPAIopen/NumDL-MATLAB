%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% classification example for peaks data using single layer neural network.
% Here we use VarPro to solve the coupled optimization problem. Compare
% this with ESingleLayer_PeaksNewtonCG

clear;

%% get peaks data
np = 4000;  % num of points sampled
nc = 5;     % num of classes
ns = 256;   % length of grid

[Y, C] = setupPeaks(np, nc, ns);
[Yv, Cv] = setupPeaks(2000, nc, ns);
[Yv, Cv] = setupPeaks(2000, nc, ns);


nf = size(Y,1);
nc = size(C,1);
%% optimize
% m = 640/20;
m = 40;
K0 = randn(m,nf);
b0 = randn(1);
act = @tanhActivation;


%% optimize
paramCl = struct('maxIter',20,'tolCG',1e-5,'maxIterCG',30,'out',-1);
paramSL = struct('act',act);
paramRegW = struct('L',1,'lambda',1e-10);
fctn = @(x,varargin) singleLayerNNVarProObjFun(x,Y,C,m,paramCl,paramSL,[],paramRegW);
x0   = [K0(:); b0(:);];

%% use steepest descent to get a starting guess
param = struct('maxIter',10,'maxStep',1);
x0 = steepestDescent(fctn,x0,param);
%% switch to newtoncg 
param = struct('maxIter',20,'tolCG',1e-6,'maxIterCG',10);
xOpt = newtoncg(fctn,x0(:),param);

%%
KOpt = reshape(xOpt(1:m*nf),m,nf);
bOpt = xOpt(m*nf+1);
Z = singleLayer(KOpt,bOpt,Y,paramSL);
fcl = @(W,varargin) softMax(W,Z,C);
WOpt = newtoncg(fcl,zeros(nc*(m+1),1),paramCl);
WOpt = reshape(WOpt,nc,[]);
%%

St = WOpt*[Z; ones(1,np)];
Sv = WOpt*[singleLayer(KOpt,bOpt,Yv,paramSL); ones(1,size(Yv,2))];
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
title(sprintf('train %1.2f%% val %1.2f%%',trainErr,valErr));


