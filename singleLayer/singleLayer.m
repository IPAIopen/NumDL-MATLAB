%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [Z,JKtmv,Jbtmv,JYtmv,JKmv,Jbmv,JYmv] = singleLayer(K,b,Y,param)
%
% Computes the Z = sigma(K*Y+b) and functions for computing J'*W and J*V
%
% Inputs:
%   K   - transformation matrix nf x m
%   b   - bias, scalar
%   Y   - input features, n x nf
%
% Output:
%   Z    - transformed features
%   JKtmv - function handle for (J_K Z)'*W
%   Jbtmv - function handle for (J_b Z)'*b
%   JYtmv - function handle for (J_Y Z)'*Y
%   JKmv  - function handle for (J_K Z)*V_K
%   Jbmv  - function handle for (J_b Z)*V_b
%   JYmv  - function handle for (J_Y Z)*V_Y

function[Z,JKtmv,Jbtmv,JYtmv,JKmv,Jbmv,JYmv] = singleLayer(K,b,Y,param)

if nargin==0
    runMinimalExample;
    return;
end

[nf,n] = size(Y);
m      = size(K,1);

if exist('param','var')
    if isfield(param,'act')
        act = param.act;
    else
        act = param;
    end
else
    act = @tanhActivation;
end

[Z,dA]  = act( K*Y+b);

JKtmv = @(W) (reshape(W,m,n).*dA)*Y';
Jbtmv = @(W) sum(sum(reshape(W,m,n).*dA));
JYtmv = @(W) K'*(reshape(W,m,n).*dA);

JKmv = @(VK) dA .* (reshape(VK,m,nf)*Y);
Jbmv = @(Vb) dA .* Vb;
JYmv = @(VY) dA .* (K*reshape(VY,nf,n));

function runMinimalExample

n  = 10;
nf = 7;
m  = 4;

K  = randn(m,nf);
Y  = randn(nf,n);
b  = randn();
Zt = feval(mfilename,K,b,Y);

dK = randn(m,nf);

tt = linspace(-1,1,51);
tb = linspace(-1,1,31);
for k=1:numel(tt)
    for j=1:numel(tb)
        F(k,j) = norm(Zt- feval(mfilename,K+tt(k)*dK,b+tb(j),Y));
    end
end
figure(1); clf;
contour(F,'linewidth',2)
set(gca,'FontSize',20);
xlabel('bias');
ylabel('kernel');
