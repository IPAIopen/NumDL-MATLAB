%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% Example: Extreme learning for MNIST
%
close all; clear all; clc;

%%
[Y,C,Yv,Cv] = setupMNIST(50000,10000);
%% optimize
m  = 1530;
nf = size(Y,1);
nc = size(C,1);

KOpt = randn(m,nf)/sqrt(nf*m);
bOpt = randn();
WOpt = randn(nc,m+1)/sqrt(nc*m);

W0 = WOpt;
Z = singleLayer(KOpt,bOpt,Y);
paramReg = struct('L',speye(numel(W0)),'lambda',1e-8);
fctn = @(x,varargin) classObjFun(x,Z,C,paramReg);
param = struct('maxIter',20,'maxStep',1,'maxIterCG',30,'tolCG',1e-2);
WOpt = newtoncg(fctn,WOpt(:),param);
%%
WOpt = reshape(WOpt,nc,m+1);
Strain = WOpt*padarray(Z,[1,0],1,'post');
S      = WOpt*padarray(singleLayer(KOpt,bOpt,Yv),[1,0],1,'post');
% the probability function
htrain = exp(Strain)./sum(exp(Strain),1);
h      = exp(S)./sum(exp(S),1);

% Find the largesr entry at each row
[~,ind] = max(h,[],1);
Cvpred = zeros(size(Cv));
Ind = sub2ind(size(Cv),ind,1:size(Cv,2));
Cvpred(Ind) = 1;
[~,ind] = max(htrain,[],1);
Cpred = zeros(size(C));
Ind = sub2ind(size(Cpred),ind,1:size(Cpred,2));
Cpred(Ind) = 1;

trainErr = 100*nnz(abs(C-Cpred))/2/nnz(C);
valErr   = 100*nnz(abs(Cv-Cvpred))/2/nnz(Cv);
fprintf('Testing    Error %3.2f%%\n',trainErr);
fprintf('Validation Error %3.2f%%\n',valErr);