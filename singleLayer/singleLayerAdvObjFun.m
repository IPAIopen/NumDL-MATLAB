%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [Ec,dE,H] = singleLayerAdvObjFun(Y,K,b,W,C)
%
% objective function for adversarial training
%
% Inputs:
%
%  Y          - input features (to be optimized here)
%  K          - parameters of transformation
%  b          - weights of bias
%  W          - weights of classifier
%  C          - desired class
%  paramSL    - struct, paramters for single layer
%  paramRegKb - struct, parameter describing regularizer for Kb
%
% Outputs:
%
%  Ec - loss function
%  dE - gradient w.r.t. Y
%  H  - approx Hessian
function [Ec,dE,H] = singleLayerAdvObjFun(Y,K,b,W,C,paramSL,paramRegKb)

if nargin==0
    runMinimalExample;
    return
end

if not(exist('paramSL','var')) || isempty(paramSL)
    paramSL = @tanhActivation;
end

n = size(C,2);
nf = size(K,2);
Y = reshape(Y,nf,n);

% evaluate layer
[Z,~,~,JYt,~,~,JY] = singleLayer(K,b,Y,paramSL);

% compute cross entropy
[Ec,~,~,dEZ,d2EY] = softMax(W,Z,C);

% add regularizer
if not(exist('paramRegKb','var')) || not(isstruct(paramRegKb))
    dS = 0; d2S = @(x) 0;
else
    [Sc,dS,d2S] = genTikhonov(Y,paramRegKb);
    Ec = Ec + Sc;
end

if nargout>1
    dEY = JYt(dEZ);
    dE  = dEY(:) + dS;
end

if nargout>2
    mat = @(Y) reshape(Y,n,nf);
    vec = @(Y) Y(:);
    H = @(Y) vec( JYt(d2EY(JY(mat(Y)))) + d2R(x));
end

function runMinimalExample
n     = 50; nf = 50; nc = 3; m  = 40;
Wtrue = randn(nc,m+1);
Ktrue = randn(m,nf);
btrue = .1;
Ytrue     = randn(nf,n);
paramSL.act = @sinActivation;
paramReg = struct('L', speye(numel(Ytrue)),'lambda',.1);

Cobs  = exp(Wtrue*padarray(singleLayer(Ktrue,btrue,Ytrue,paramSL),[1 0],1,'post'));
Cobs  = Cobs./sum(Cobs,1);

Y0     = randn(nf*n,1);
[E,dE] = feval(mfilename,Y0,Ktrue,btrue,Wtrue,Cobs,paramSL,paramReg);
dY = randn(size(Y0));
for k=1:20
    h  = 2^(-k);
    Et = feval(mfilename,Y0+h*dY,Ktrue,btrue,Wtrue,Cobs,paramSL,paramReg);
    
    err1 = norm(E-Et);
    err2 = norm(E+h*dE'*dY-Et);
    fprintf('%1.2e\t%1.2e\t%1.2e\n',h,err1,err2);
end

