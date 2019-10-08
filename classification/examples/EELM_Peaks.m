%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% Example: Extreme learning for Peaks example
%
close all; clear all; clc;
rng(42)
%% get peaks data
np = 8000;  % num of points sampled
nc = 5;     % num of classes
ns = 256;   % length of grid

[Y, C] = setupPeaks(np, nc, ns);

numTrain = size(Y, 2)*0.80;
idx = randperm(size(Y,2));
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
m = 200;
KOpt = randn(m,nf);
bOpt = randn(1);
W0   = randn(nc,m+1);

%% compare nonlinearities
figure(1); clf;
subplot(2,2,1);
Z1 = sin(KOpt*Y+bOpt);
Z2 = tanh(KOpt*Y+bOpt);
Z3 = max(0,KOpt*Y+bOpt);
semilogy(svd(Z1),'linewidth',3);
hold on;
semilogy(svd(Z2),'linewidth',3);
semilogy(svd(Z3),'linewidth',3);
legend('sin','tanh','relu');
title('singular values')
set(gca,'FontSize',20)
%% optimize
relu = @(x) max(x,0);
acts = {@sin,@tanh,relu};

for k=1:numel(acts)
    act  = acts{k};
    Z    = act(KOpt*Y+bOpt);
    paramRegW = struct('L',speye(numel(W0)),'lambda',1e-3);
    fctn = @(x,varargin) classObjFun(x,Z,C,paramRegW);
    param = struct('maxIter',30,'maxStep',1,'tolCG',1e-3,'maxIterCG',100);
    WOpt = newtoncg(fctn,W0(:),param);
    %%
    WOpt = reshape(WOpt,nc,m+1);
    Strain = WOpt*padarray(Z,[1,0],1,'post');
    S      = WOpt*padarray(act(KOpt*YTest+bOpt),[1,0],1,'post');
    htrain = exp(Strain)./sum(exp(Strain),1);
    h      = exp(S)./sum(exp(S),1);
    
    % Find the largesr entry at each row
    [~,ind] = max(h,[],1);
    Cv = zeros(size(CTest));
    Ind = sub2ind(size(Cv),ind,1:size(Cv,2));
    Cv(Ind) = 1;
    [~,ind] = max(htrain,[],1);
    Cpred = zeros(size(C));
    Ind = sub2ind(size(Cpred),ind,1:size(Cpred,2));
    Cpred(Ind) = 1;
    
    trainErr = 100*nnz(abs(C-Cpred))/2/nnz(C);
    valErr   = 100*nnz(abs(Cv-CTest))/2/nnz(Cv);
    %%
    x = linspace(-3,3,201);
    [Xg,Yg] = ndgrid(x);
    Z = WOpt * padarray(act(KOpt*[vec(Xg)'; vec(Yg)']+bOpt),[1,0],1,'post');
    h      = exp(Z)./sum(exp(Z),1);
    
    [~,ind] = max(h,[],1);
    Cpred = zeros(5,numel(Xg));
    Ind = sub2ind(size(Cpred),ind,1:size(Cpred,2));
    Cpred(Ind) = 1;
    img = reshape((1:5)*Cpred,size(Xg));
    %%
    figure(1);
    subplot(2,2,1+k)
    imagesc(x,x,img')
    title(sprintf('%s - train %1.2f%% val %1.2f%%',func2str(act),trainErr,valErr));    
end
%%
for k=1:4
    subplot(2,2,k)
    set(gca,'FontSize',20)
end


