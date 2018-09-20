%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% Example for 1D convolution with FFTs

close all; clear all; clc;

% stencil
n     = 16;
theta = rand(3,1);
K     = full(spdiags(ones(n,1)*flipud(theta)',-1:1,n,n));
K(1,end) = theta(3);
K(end,1) = theta(1)

%% compute eigenvalues using FFT
eigK  = eig(K);
eigKt = fft(K(:,1));
figure(1); clf
plot(real(eigK),imag(eigK),'or');
hold on;
plot(real(eigKt),imag(eigKt),'.b');
xlabel('real')
ylabel('imag');
set(gca,'FontSize',20);

%%
x = linspace(0,1,n)';
y = cos(2*pi*x);

z1 = K*y;
z2 = real(ifft(eigKt.*fft(y)));

figure(2); clf;
subplot(1,3,1)
plot(x,y,'linewidth',2);
subplot(1,3,2);
plot(x,z1,'linewidth',2);
hold on;
plot(x,z2,'linewidth',2);
legend('z=K*y','z=ifft(fft(y).*lam)')
subplot(1,3,3);
semilogy(x,abs(z1-z2),'linewidth',2);
title('error')
for k=1:3; subplot(1,3,k); set(gca,'FontSize',20); end;

%%
z1 = K'*y;
z2 = real(fft(eigKt.*ifft(y)));
norm(z1-z2)

%% check derivatives
y   = randn(n,1);
th0 = randn(3,1);


%% code for generating first column
sK  = numel(theta); center = (sK-1)/2+1;
Ku = zeros(16,1); Ku(1:center) = theta(center:end);
Ku(end-(center-2):end) = theta(1:center-1);
%% using circshift
Kt = zeros(16,1); Kt(1:sK)=theta; circshift(Kt,1-center)

