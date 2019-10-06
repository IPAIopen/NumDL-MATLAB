%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% plot the loss landscape for the softmax loss and a single layer neural
% network
close all; clear all; clc;

rng(2)
n  = 50; nf = 50; nc = 3; m  = 40;
Wtrue = randn(nc,m+1);
Ktrue = randn(m,nf);
btrue = .1;

Y     = randn(nf,n);
Cobs  = exp(Wtrue* padarray(singleLayer(Ktrue,btrue,Y),[1 0],1,'post'));
Cobs  = Cobs./sum(Cobs,1);

%%
dW = randn(nc,m+1);
dK = randn(m,nf);

[tW,tK] = ndgrid(linspace(-1,1,41));
E = 0*tW;
for i=1:size(tW,1)
    for j=1:size(tW,2)
        Zt = singleLayer(Ktrue+tK(i,j)*dK,btrue,Y);
        E(i,j)=softMax(Wtrue+tW(i,j)*dW,Zt,Cobs);
        
    end
end

%%
figure(1); clf;
contour(tW,tK,E,'lineWidth',2)
xlabel('W + tW*dW')
ylabel('K + tK*dK')
set(gca,'FontSize',20)

%%
figure(2); clf;
surfc(tW,tK,E,'lineWidth',2)
xlabel('W + tW*dW')
ylabel('K + tK*dK')
set(gca,'FontSize',20)