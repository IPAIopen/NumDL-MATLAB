%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [Z,Jthtmv,Jbtmv,JYtmv,Jthmv,Jbmv,JYmv] = singleConvLayer(theta,b,Y,param)
%
% Computes the Z = sigma(K*Y+b) and functions for computing J'*W and J*V
%
% Inputs:
%   K     - theta, weights for convolution matrix
%   b     - bias, scalar
%   Y     - input features, n x nf
%   param - struct, description of activation and conv operator
%
% Output:
%   Z    - transformed features
%   Jthtmv - function handle for (J_th Z)'*W
%   Jbtmv - function handle for (J_b Z)'*b
%   JYtmv - function handle for (J_Y Z)'*Y
%   Jthmv  - function handle for (J_th Z)*V_K
%   Jbmv  - function handle for (J_b Z)*V_b
%   JYmv  - function handle for (J_Y Z)*V_Y
function[Z,Jthtmv,Jbtmv,JYtmv,Jthmv,Jbmv,JYmv] = singleConvLayer(theta,b,Y,param)


if nargin==0
    runMinimalExample;
    return;
end

if isfield(param,'act')
    act = param.act;
else
    act = @tanhActivation;
end
kernel = param.kernel;
K = getOp(kernel,theta);
b = reshape(b,1,1,[]);


[Z,dA]  = act( K*Y+b);

if nargout>1
    szZ = size(Z);
    Jthtmv = @(W) JthetaTmv(kernel,reshape(W,szZ).*dA,[],Y);
    Jbtmv = @(W) sum(sum(sum(reshape(W,size(dA)).*dA,4),2),1);
    JYtmv = @(W) K'*(reshape(W,size(dA)).*dA);
    
    Jthmv = @(VK) dA .* (reshape(Jmv(VK),m,nf)*Y);
    Jbmv = @(Vb) dA .* reshape(Vb,1,1,[]);
    szY = size(Y);
    JYmv = @(VY) dA .* (K*reshape(VY,szY));
end


function runMinimalExample

nImg   = [18 16];
sK     = [3 3 2 4];
n      = 10;
kernel = convFFT(nImg,sK);

param  = struct('kernel',kernel);

theta  = randn(sK);
Y  = randn([nImg sK(3) n]);
b  = randn(sK(4),1);
Zt = feval(mfilename,theta,b,Y,param);

dK = randn(sK);

tt = linspace(-1,1,51);
tb = linspace(-1,1,31);
for k=1:numel(tt)
    for j=1:numel(tb)
        F(k,j) = norm(vec(Zt- feval(mfilename,theta+tt(k)*dK,b+tb(j),Y,param)));
    end
end
figure(1); clf;
contour(F,'linewidth',2)
xlabel('bias');
ylabel('kernel');
set(gca,'fontsize',20)
