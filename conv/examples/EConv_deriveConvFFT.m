%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% Here, we derive a matrix-free implementation of a 1D convolution with
% FFTs
%

clc; clear;
m     = 6;           % number of cells in grid
theta = [1 2 3]';    % stencil

% build convolution operator
K      = spdiags(ones(m,1)*flipud(theta)',-1:1,m,m);
% periodic boundary conditions
K(1,end) = theta(3);
K(end,1) = theta(1);
K = full(K)

%% verify that eigenvalues can be obtained by using fft (up to ordering)
lam = fft(K(:,1));
lamt = eig(K)

[lam lamt]

%% verify that convolution can be computed as F^{-1)(lam .* F(y))
y = randn(m,1);
errKy = norm(K*y - ifft(lam.*fft(y)))
%% verify equation for transpose of convolution operator
errKTy = norm(K'*y - ifft(conj(lam).*fft(y)))

%% verify that first column of K can be computed using circshift
theta = [1;2;3;];
center = (numel(theta)+1)/2;
Ku = circshift([theta;zeros(m-numel(theta),1)],1-center);
errKu = norm(K(:,1)-Ku)