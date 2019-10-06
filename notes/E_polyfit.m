%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% This driver generates example connecting generalization to overfitting in
% polynomial interpolation.
%
close all; clear all;


xf = linspace(0,1,101);
xt = linspace(0,1,10);
ft = sin(pi*xt.^2)+5e-2*randn(1,10);

xv = 0.97;
fv = sin(pi*xv.^2);

fig1 = figure(1); clf;
fig1.Name ='data'
p1 = plot(xt,ft,'.','MarkerSize',30)
hold on;
plot(xv,fv,'.','MarkerSize',30)
l = legend('training data','validation data');
l.Location = 'NorthWest';

fig2 = figure(2); clf;
fig2.Name ='overfit'
p = polyfit(xt,ft,numel(xt)+1);
pf = polyval(p,xf);
plot(xt,ft,'.','MarkerSize',40)
hold on;
plot(xv,fv,'.','MarkerSize',40)
plot(xf,pf,'-','LineWidth',2,'Color',p1.Color)
l = legend('training data','validation data','model');
l.Location = 'NorthWest';

fig3 = figure(3); clf;
fig3.Name ='underfit'
p = polyfit(xt,ft,2);
pf = polyval(p,xf);
plot(xt,ft,'.','MarkerSize',40)
hold on;
plot(xv,fv,'.','MarkerSize',40)
plot(xf,pf,'-','LineWidth',2,'Color',p1.Color)

l = legend('training data','validation data','model');
l.Location = 'NorthWest';

return;
figDir = '/Users/lruthot/Dropbox/Projects/NumDL-CourseNotes/images/'

for k=1:3
    fig = figure(k)
    axis([0 1 -.3 1.4])
     axis square 
    set(gca,'FontSize',30)
    printFigure(gcf,fullfile(figDir,['generalize_' fig.Name '.png']))
end

