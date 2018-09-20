%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% Training a ResNet for solving the Peaks example
%

clear all;
[Y,C]    = setupPeaks();
fig      = figure(1); 
fig.Name = 'exResNet_Peaks: True function';

%% choose an activation function
paramResnet.act = @sinActivation;
% param.act = @tanhActivation;
% param.act = @reluActivation;

%% parameters for the ResNet
nc = 8; % width of the ResNet 
T  = 4;  % final time of the ResNet
nt = 16; % number of layers

%% set this flag to true to initialize the ResNet with an
% anti-symmetric weight matrix for which forward Euler is uncoditionally
% unstable.
startUnstable = false; 

%% build the description of the ResNet (i.e., specify P for each layer) and
% initialize the weights (i.e., K and b)
n = nc*ones(2,nt);
n(1)=2;
 for i=1:size(n,2)
    if i<3
        Ki = randn(n(2,i),n(1,i))/sqrt(prod(n(:,i)));
    end
    if i>1 && startUnstable
          Ki = Ki-Ki';
    end
    K{i} = Ki;
    if n(1,i) ~= n(2,i)
        P{i} = opZero(n(2,i),n(1,i));
    else
        P{i} = opEye(n(1,i));
    end
    b{i} = zeros(n(2,i),1);
end
N = length(P);
paramResnet.P = P;
paramResnet.h= T/nt;
paramResnet.n = n;

%% train the network
Kb = [cell2vec(K); cell2vec(b)];
W  = randn((n(end,end)+1)*size(C,1),1);
x0 = [Kb;W];

%% specify regularizer
LKb = speye(numel(Kb));
LW  = speye(numel(W));
paramReg.L = blkdiag(LKb,LW);
paramReg.lambda = 1e-5;

fctn = @(x,varargin) ResNetObjFun(x,Y,C,numel(Kb),paramResnet,paramReg);

paramOpt = struct('maxIter',50,'maxStep',.1);
xsol = newtoncg(fctn,x0,paramOpt);

%% show results
x = linspace(-3,3,201);
[Xg,Yg] = ndgrid(x);

KbOpt = xsol(1:numel(Kb));
WOpt  = reshape(xsol(numel(Kb)+1:end),size(C,1),[]);
[Yo,~,Yall] = ResNetForward(KbOpt,[Xg(:)';Yg(:)'],paramResnet);
Z = WOpt*padarray(Yo,[1 0],1,'post');
h      = exp(Z)./sum(exp(Z),1);

[~,ind] = max(h,[],1);
Cpred = zeros(5,numel(Xg));
Ind = sub2ind(size(Cpred),ind,1:size(Cpred,2));
Cpred(Ind) = 1;
img = reshape((1:5)*Cpred,size(Xg));

fig= figure(2); clf
fig.Name = 'exResNet_Peaks: Results';
imagesc(x,x,img')

