%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [Ec,dE,H] = singleLayerNNVarProObjFun(x,Y,C,m,paramCl,paramSL,paramRegKb,paramRegW)
%
% evaluates single layer and computes cross entropy, gradient and approx. Hessian
%
% Let x = [K(:);b(:);W(:)], we compute
%
% E(x) = E(Z*W,C),   where Z = activation(Y*K+b)
%
% Inputs
%
%   x          - current iterate, x=[K(:);b(:);W(:)]
%   Y          - input features
%   C          - class probabilities
%   m          - size(K,2), used to split x correctly
%   paramCl    - struct, paramters for newtoncg used to classify
%   paramSL    - struct, paramters for single layer
%   paramRegKb - struct, parameter describing regularizer for K and b
%   paramRegW  - struct, parameter describing regularizer for W
%
% Output:
%
%   Ec - current value of loss function
%   dE - gradient w.r.t. K,b,W, vector
%   H  - approximate Hessian, H=J'*d2ES*J, function_handle
function [Ec,dE,H] = singleLayerNNVarProObjFun(x,Y,C,m,paramCl,paramSL,paramRegKb,paramRegW)

if nargin==0
    runMinimalExample;
    return;
end

if not(exist('paramSL','var')) || isempty(paramSL)
    paramSL = @tanhActivation;
end

[nf,n] = size(Y);
nc     = size(C,1);

% split x into K,b
x = x(:);
K = reshape(x(1:nf*m),m,nf);
b = x(nf*m+1);

% evaluate layer
[Z,JKt,Jbt,~,JK,Jb,~] = singleLayer(K,b,Y,paramSL);

% solve classification problem
if exist('paramRegW','var')
    fctn = @(W,varargin) classObjFun(W,Z,C,paramRegW);
else
    fctn = @(W,varargin) softMax(W,Z,C);
end
WOpt = newtoncg(fctn,zeros(nc*(m+1),1),paramCl);

% call cross entropy
[Ec,~,~,dEZ,d2EZ] = softMax(WOpt,Z,C);

% regularizer
if not(exist('paramRegKb','var')) || not(isstruct(paramRegKb))
    Sc = 0; dS = 0; d2S = @(x) 0;
else
    [Sc,dS,d2S] = genTikhonov(x,paramRegKb);
end

Ec = Ec + Sc;
if nargout>1
    dEK = JKt(dEZ);
    dEb = Jbt(dEZ);
    
    dE  = [dEK(:); dEb(:);];
    dE  = dE + dS;
end

if nargout>2
    szK = [size(K,1) size(K,2)];
    H = @(x) HessMat(x,szK,JK,Jb,JKt,Jbt,d2EZ,d2S);
end

function Hx = HessMat(x,szK,JK,Jb,JKt,Jbt,d2EY,d2S)
nK = prod(szK);

% split x
xK = x(1:nK);
xb = x(nK+1);

% compute Jac*x
JKbx = JK(reshape(xK,szK)) + Jb(xb);
tt   = d2EY(JKbx);
Hx1  = [reshape(JKt(tt),[],1); Jbt(tt) ];

% stack result
Hx = Hx1(:) + d2S(x);


function runMinimalExample

n  = 50; nf = 50; nc = 3; m  = 40;
Wtrue = randn(nc,m+1);
Ktrue = randn(m,nf);
btrue = .1;

Y     = randn(nf,n);
Cobs  = exp(Wtrue*padarray(singleLayer(Ktrue,btrue,Y),[1 0],1,'post'));
Cobs  = Cobs./sum(Cobs,1);

x0     = [Ktrue(:);btrue;];
x0 = randn(size(x0));
paramCl = struct('maxIter',100,'maxIterCG',100,'tolCG',1e-4,'out',0);
paramRegW = struct('L',speye(numel(Wtrue)),'lambda',1e-2);
[E,dE] = feval(mfilename,x0,Y,Cobs,m,paramCl);
dK = randn(nf*m,1);
db = randn();
dx     = 100*[dK;db];
for k=1:20
    h  = 2^(-k);
    Et = feval(mfilename,x0+h*dx,Y,Cobs,m,paramCl);
    
    err1 = norm(E-Et);
    err2 = norm(E+h*dE'*dx-Et);
    fprintf('%1.2e\t%1.2e\t%1.2e\n',h,err1,err2);
end







