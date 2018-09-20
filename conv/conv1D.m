%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [KYmv,KtYmv,Jmv,Jtmv] = conv1D(nnImgf,theta,Y)
%
% computes 1D convolutions using FFT
%
%  Z = K(theta)* Y,
%
% where size(Y)=nImg x n and size(K)= nImg x nf.
%
% Input:
%   nImg  - number of grid point (i.e., number of features)
%   theta - stencil (assumed to be odd number of elements)
%   Y     - features (needed only for Jacobians)
%
% Output
%   KYmv  - function handle for Y -> K(theta)*Y
%   KtYmv - function handle for Y -> K(theta)'*Y
%   Jmv   - function handle for v -> J(K(theta)*Y) * v
%   Jtmv  - function handle for w -> J(K(theta)*Y)'* w

function [KYmv,KtYmv,Jmv,Jtmv] = conv1D(nImg,theta,Y)

if nargin==0
    testThisMethod
    return;
end
lam   = fft(getK1(theta,nImg));
sdiag = @(v) spdiags(v(:),0,numel(v),numel(v));
KYmv  = @(Y) real(ifft(sdiag(lam)*fft(Y))); 
KtYmv = @(Y) real(fft(sdiag(lam)*ifft(Y)));

% Jacobians
if nargout>2
    iFy = fft(Y);
    q   = getK1(1:numel(theta),nImg);
    I   = find(q);
    J   = q(I);
    Q   = sparse(I,J,ones(numel(theta),1),nImg,numel(theta));
    
    Jmv  = @(v) real(ifft(sdiag(fft(Q*v))*iFy)); 
    Jtmv = @(w) real(Q'*fft(sum(iFy.*ifft(w),2)));
end


% ---- helper functions ----
function K1 = getK1(theta,m)
% builds first column of convolution operator K(theta)
center = (numel(theta)+1)/2;
K1 = circshift([theta(:);zeros(m-numel(theta),1)],1-center);

% ----- test function -----
function testThisMethod
nf    = 16;
n     = 10;
theta = randn(3,1);
K     = full(spdiags(ones(nf,1)*flipud(theta)',-1:1,nf,nf));
K(1,end) = theta(3);
K(end,1) = theta(1);
Y = randn(nf,n);

[KYmv,KYtmv,Jmv,Jtmv] = feval(mfilename,nf,theta,Y);

T1 = K*Y;
T2 = KYmv(Y);
fprintf('error for K*Y:   %1.2e\n',norm(T1-T2))

T1 = K'*Y;
T2 = KYtmv(Y);
fprintf('error for K''*Y:  %1.2e\n',norm(T1-T2))

% derivative check
dth = randn(size(theta));
KYmv2 = feval(mfilename,nf,theta+dth,Y);
T1 = KYmv2(Y);
T2 = KYmv(Y) + Jmv(dth);
fprintf('error for J*v:   %1.2e\n',norm(T1-T2));

% adjoint check
dZ = randn(size(Y));
T1 = sum(sum(dZ.*Jmv(dth)));
T2 = sum(sum(dth.*Jtmv(dZ)));
fprintf('adjoint error:   %1.2e\n',norm(T1-T2));

