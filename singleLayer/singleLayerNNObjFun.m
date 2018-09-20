%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [Ec,dE,H] = singleLayerNNObjFun(x,Y,C,m)
%
% evaluates single layer and computes cross entropy, gradient and approx. Hessian
%
% Let x = [K(:);b(:);W(:)], we compute
%
% E(x) = E(Z*W,C),   where Z = activation(Y*K+b)
%
% Inputs:
%
%   x - current iterate, x=[K(:);b(:);W(:)]
%   Y - input features
%   C - class probabilities
%   m - size(K,2), used to split x correctly
%   paramSL - struct, description of single layer. Fields:
%             act - activation function (default: @tanhActivation)
%   paramReg- struct, description of regularizer. See genTikhonov.m
%
% Output:
%
%   Ec - current value of loss function
%   dE - gradient w.r.t. K,b,W, vector
%   H  - approximate Hessian, H=J'*d2ES*J, function_handle
function [Ec,dE,H] = singleLayerNNObjFun(x,Y,C,m,paramSL,paramReg)


if nargin==0
    runMinimalExample
    return;
end

if not(exist('paramSL','var')) || isempty(paramSL)
    paramSL = @tanhActivation;
end


[nf,~] = size(Y);
nc     = size(C,1);

% split x into K,b,W
x = x(:);
K = reshape(x(1:nf*m),m,nf);
b = x(nf*m+1);
W = reshape(x(nf*m+2:end),nc,[]);

% evaluate layer
[Z,JKt,Jbt,~,JK,Jb,~] = singleLayer(K,b,Y,paramSL);

% compute cross entropy
[Ec,dEW,d2EW,dEZ,d2EZ] = softMax(W,Z,C);

% add regularizer
if not(exist('paramReg','var')) || not(isstruct(paramReg))
    dS = 0; d2S = @(x) 0;
else
    [Sc,dS,d2S] = genTikhonov(x,paramReg);
    Ec = Ec + Sc;
end

if nargout>1
    dEK = JKt(dEZ);
    dEb = Jbt(dEZ);
    
    dE  = [dEK(:); dEb(:); dEW(:)];
    dE  = dE + dS;
end


if nargout>2
    szK = [size(K,1) size(K,2)];
    H = @(x) HessMat(x,szK,JK,Jb,JKt,Jbt,d2EW,d2EZ,d2S);
end

function Hx = HessMat(x,szK,JK,Jb,JKt,Jbt,d2EW,d2EY,d2S)
nK = prod(szK);

% split x
xK = x(1:nK);
xb = x(nK+1);
xW = x(nK+2:end);

% compute Jac*x
JKbx = JK(reshape(xK,szK)) + Jb(xb);
tt   = d2EY(JKbx);
Hx1  = [reshape(JKt(tt),[],1); Jbt(tt) ];

Hx2  = d2EW(xW);

% stack result
Hx = [Hx1(:); Hx2(:)] + d2S(x);


function runMinimalExample
n  = 50; nf = 50; nc = 3; m  = 40;
Wtrue = randn(nc,m+1);
Ktrue = randn(m,nf);
btrue = .1;

Y     = randn(nf,n);
Cobs  = exp(Wtrue*padarray(singleLayer(Ktrue,btrue,Y),[1,0],1,'post'));
Cobs  = Cobs./sum(Cobs,1);

x0     = [Ktrue(:);btrue; Wtrue(:)];
x0 = randn(size(x0));
[E,dE] = feval(mfilename,x0,Y,Cobs,m);
dK = randn(nf*m,1);
db = randn();
dW = randn((m+1)*nc,1);
dx     = [dK;db;dW];
for k=1:20
    h  = 2^(-k);
    Et = feval(mfilename,x0+h*dx,Y,Cobs,m);
    
    err1 = norm(E-Et);
    err2 = norm(E+h*dE'*dx-Et);
    fprintf('%1.2e\t%1.2e\t%1.2e\n',h,err1,err2);
end






