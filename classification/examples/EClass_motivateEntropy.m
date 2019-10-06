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

[W1,W2] = meshgrid(linspace(-7,7,201));

F = zeros(101,101);
F2 = F;
for i=1:size(W1,1)
    for j=1:size(W1,1)
        Hp = 1./(1+exp(Y*[W1(i,j);W2(i,j)]));
        Hp= [Hp 1-Hp];
        Cp = Hp./(sum(Hp,2));
       F(i,j) = 0.5*norm(Cp-C)^2/nex;
       
       F2(i,j) = -sum(sum(C.*log(Cp)))/nex;
    end
end

fig = figure(1);clf; 
fig.Name = 'Frobenius'
% subplot(1,2,1)
contour(W1,W2,F,50,'linewidth',2)
% hold on;
% [mx,idx] = min(F(:));
% plot(W1(idx),W2(idx),'.r','MarkerSize',60)
title('Frobenius Norm')
% colorbar
axis equal tight

fig = figure(2);clf; 
fig.Name = 'CrossEntropy'
% subplot(1,2,2)
contour(W1,W2,F2,50,'linewidth',2)
% hold on;
% [mx,idx] = min(F2(:));
% plot(W1(idx),W2(idx),'.r','MarkerSize',60)
title('cross entropy')
axis equal tight
% colorbar;

figDir = '/Users/lruthot/Dropbox/Projects/NumDL-CourseNotes/images/'

for k=1:2
    fig = figure(k)
    title([])
    axis equal tight off
    set(gca,'FontSize',20)
    printFigure(gcf,fullfile(figDir,['Class_' fig.Name '.png']))
end

