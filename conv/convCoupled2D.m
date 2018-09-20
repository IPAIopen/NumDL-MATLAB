%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [YKmv,YKtmv,Jmv,Jtmv] = convCoupled2D(nImg,sTheta,theta,Y)
%
% computes the coupled convolution of multi-channel images Y.
%
%
% Input:
%  nImg   -  number of pixels, e.g., nImg = [16,16];
%  sTheta -  size of kernel, e.g., sTheta = [3,3,4,6] for 3x3 convolutions
%            applied to 4 input channels giving 6 output channels
%  theta  -  weights
%  Y      -  feature matrix, only needed for derivative computation
%
% Output:
%  YKmv   -  function handle for computing Y -> K(theta)*Y
%  YKtmv  -  function handle for computing Y -> K(theta)'*Y
%  Jmv    -  function handle for computing v -> Jac*v
%  Jtmv   -  function handle for computing w -> Jac'*w
function [KYmv,KYtmv,Jmv,Jtmv] = convCoupled2D(nImg,sTheta,theta,Y)


if nargin==0
    testThisMethod;
    return
end

KYmv  = @(Y) Amv(nImg,sTheta,theta,Y);
KYtmv = @(Y) Atmv(nImg,sTheta,theta,Y);
if nargout>2
    Jmv  = @(v) Amv(nImg,sTheta,v,Y);
    Jtmv = @(Z) JthetaTmv(nImg,sTheta,Y,Z);
end


function Z = Amv(nImg,sTheta,theta,Y)
% compute convolution
nex   = size(Y,4);

Z     = zeros([nImg sTheta(4) nex],'like',Y);
S     = reshape(fft2(getK1(theta,nImg,sTheta)),[nImg sTheta(3:4)]);
Yh    = fft2(Y);
for k=1:sTheta(4)
    T  = S(:,:,:,k) .* Yh;
    Z(:,:,k,:)  = sum(T,3);
end
Z = real(ifft2(Z));

function Y = Atmv(nImg,sTheta,theta,Z)
% compute transpose of convolution

nex =  size(Z,4);
Y = zeros([nImg sTheta(3) nex],'like',Z);
S   = reshape(fft2(getK1(theta,nImg,sTheta)),[nImg sTheta(3:4)]);

Zh = ifft2(Z);
for k=1:sTheta(3)
    Sk = squeeze(S(:,:,k,:));
    Y(:,:,k,:) = sum(Sk.*Zh,3);
end
Y = real(fft2(Y));


function dtheta = JthetaTmv(nImg,sTheta,Y,Z)
% compute Jac'*Z

dth1 = zeros(prod(sTheta(1:3)),sTheta(4),'like',Y);
Yh   = permute(fft2(Y),[1 2 4 3]);
Zh   = ifft2(Z);

% get q vector for a given row in the block matrix
v   = vec(1:prod(sTheta(1:3)));
q   = getK1(v,nImg,sTheta);

I    = find(q(:));
for k=1:sTheta(4)
    Zk = squeeze(Zh(:,:,k,:));
    tt = squeeze(sum(Zk.*Yh,3));
    tt = real(fft2(tt));
    dth1(q(I),k) = tt(I);
end
dtheta = dth1(:);

function K1 = getK1(theta,nImg,sTheta)
% compute first row of convolution matrix
theta = reshape(theta,sTheta(1),sTheta(2),[]);
center = (sTheta(1:2)+1)/2;

K1  = zeros([nImg size(theta,3)],'like',theta);
K1(1:sTheta(1),1:sTheta(2),:) = theta;
K1  = circshift(K1,1-center);

function testThisMethod

nImg   = [16 16];
sTheta = [3 3 4 6];
n      = 1;
theta  = ones(prod(sTheta),1);
Y      = randn([nImg sTheta(3) n]);


[KYmv,KYtmv,Jmv,Jtmv] = feval(mfilename,nImg,sTheta,theta,Y);
YK = KYmv(Y);
Z  = randn(size(YK),'like',YK);
YKtZ = KYtmv(Z);
t1 = sum(vec(YK.*Z));
t2 = sum(vec(YKtZ.*Y));
fprintf('adjoint  error:     %1.2e\n',norm(t1-t2))


% derivative check
dth = randn(size(theta),'like',theta);
Kmv2 = feval(mfilename,nImg,sTheta,theta+dth,Y);
T1 = Kmv2(Y);
T2 = KYmv(Y) + Jmv(dth);
fprintf('error in Jacobian: %1.2e\n',norm(vec(T1-T2)))

% adjoint check
dZ = randn(size(T2),'like',theta);
dth = randn(size(dth),'like',theta);
T1 = sum(vec(dZ.*Jmv(dth)));
T2 = sum(vec(dth(:)'*Jtmv(dZ)));
fprintf('adjoint Jacobian error:     %1.2e\n',norm(T1-T2))

