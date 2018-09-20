%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [KYmv,KYtmv,Jmv,Jtmv] = conv2D(nImg,sTheta,theta,Y)
%
% computes 2D convolutions using FFT
%
% Input: 
%   nImg   - dimension of image data
%   sTheta - stencil size
%   theta  - stencil
%   Y      - images
%
% Output:
%   KYmv  - function handle for Y -> K(theta)*Y
%   KtYmv - function handle for Y -> K(theta)'*Y
%   Jmv   - function handle for v -> J(K(theta)*Y) * v
%   Jtmv  - function handle for w -> J(K(theta)*Y)'* w
function [KYmv,KYtmv,Jmv,Jtmv] = conv2D(nImg,sTheta,theta,Y)

if nargin==0
    runMinimalExample
    return
end

rshp3D = @(Y) reshape(Y',nImg(1),nImg(2),[]);
rshp2D = @(Y) reshape(Y,prod(nImg),[]);
vec    = @(V) V(:);

lam   = fft2(getK1(theta,nImg,sTheta));
KYmv  = @(Y) rshp2D(real(ifft2(lam.*fft2(rshp3D(Y)))));
KYtmv = @(Y) rshp2D(real(fft2(lam.*ifft2(rshp3D(Y)))));

if nargout>2
    Fy = fft2(rshp3D(Y));
    q   = getK1(1:numel(theta),nImg,sTheta);
    I   = find(q);
    J   = q(I);
    Q   = sparse(I,J,ones(numel(theta),1),prod(nImg),numel(theta));
    
    Jmv  = @(v) rshp2D(real(ifft2(fft2(getK1(v,nImg,sTheta)).*Fy)));
    Jtmv = @(w) real(Q'*vec(fft2(sum(Fy.*ifft2(rshp3D(w)),3))));
end

function K1 = getK1(theta,nImg,sTheta)
theta = reshape(theta,sTheta);
K1 = zeros(nImg,'like',theta);
K1(1:sTheta(1), 1:sTheta(2)) = theta;
center = (sTheta+1)/2;
K1  = circshift(K1,1-center);

function runMinimalExample
nImg  = [16 16];
xa    = linspace(0,1,nImg(1));
ya    = linspace(0,1,nImg(2));
[X,Y] = ndgrid(xa,ya);
theta = [-1 0 1; -1 0 1; -1 0 1];
y     = X(:)+Y(:);

[KYmv,KYtmv,Jmv,Jtmv] = feval(mfilename,nImg,size(theta),theta,y);

T2 = KYmv(y);
figure(1); clf;
subplot(1,3,1);
imagesc(reshape(y,nImg));
subplot(1,3,2);
imagesc(reshape(T2,nImg));


T2 = KYtmv(y);
subplot(1,3,3);
imagesc(reshape(T2,nImg));

% derivative check
dth = randn(size(theta));
KYmv2 = feval(mfilename,nImg,size(theta),theta+dth,y);
T1 = KYmv2(y);
T2 = KYmv(y) + Jmv(dth);
fprintf('error in Jacobian: %1.2e\n',norm(T1-T2))

% adjoint check
dZ = randn(size(y));
dth = randn(size(dth));
T1 = sum(sum(dZ.*Jmv(dth)));
T2 = sum(sum(dth(:)'*Jtmv(dZ)));
fprintf('adjoint error:     %1.2e\n',norm(T1-T2))

