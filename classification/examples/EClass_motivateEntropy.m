%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% This basic example illustrates the convexity of the entropy function
% 
nex = 100;
nf  = 2;

Y   = randn(nex,nf);
C1  = sum(Y,2)>0;
C   = [C1 1-C1];

[W1,W2] = meshgrid(linspace(-2,5,101));

F = zeros(101,101);
F2 = F;
for i=1:101
    for j=1:101
        Hp = 1./(1+exp(Y*[W1(i,j);W2(i,j)]));
        Hp= [Hp 1-Hp];
        Cp = Hp./(sum(Hp,2));
       F(i,j) = 0.5*norm(Cp-C)^2/nex;
       
       F2(i,j) = -sum(sum(C.*log(Cp)))/nex;
    end
end

figure; 
subplot(1,2,1)
contour(W1,W2,F,50)
title('Frobenius Norm')
colorbar

subplot(1,2,2)
contour(W1,W2,F2,50)
title('cross entropy')
colorbar;