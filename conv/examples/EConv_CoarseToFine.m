%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% demo for prolongation of 1D convolution operators
%
clc; close all;

% number of discretization points
nc = 16;
nf = 2*nc;
hc = 1/nc;
hf = 1/nf;

% grids
x = linspace(0,1,101);
xc = linspace(0,1,nc);
xf = linspace(0,1,nf);

%% get A(hc) and A(hf)
thc = [-67; 129; -59]; 
fprintf('theta(coarse):\t[%1.3f,%1.3f,%1.3f]\n',thc);
Ac    = [1 -1 -1; 1 0 2; 1 1 -1] * diag([1/4;1/(2*hc);1/hc^2]);
beta  = Ac\thc;
fprintf('beta:         \t[%1.3f,%1.3f,%1.3f]\n',beta);
Af    = [1 -1 -1; 1 0 2; 1 1 -1] * diag([1/4;1/(2*hf);1/hf^2]);
thf   = Af*beta;
fprintf('theta(fine):  \t[%1.3f,%1.3f,%1.3f]\n',thf);
%%
f   = @(x) (cos(2*pi*x.^4))+x-.8*(x-.5).^2;
df  = @(x) x - (4*(x - 1/2).^2)/5 - 8*x.^3.*pi.*sin(2*pi*x.^4);
d2f = @(x) 9/5 - 24*x.^2.*pi.*sin(2*pi*x.^4) - 64*x.^6.*pi^2.*cos(2*pi*x.^4) - (8*x)/5;

z = @(x) beta(1)*f(x)+beta(2)*df(x)+beta(3)*d2f(x);

fig = figure(1); clf;
fig.Name = 'E15CoarseToFineConv1D';
subplot(2,2,1);
plot(x,f(x),'-b','LineWidth',2);
set(gca,'FontSize',20)
title('function,f')

subplot(2,2,2)
fig.Name = 'z';
plot(x,z(x),'-r','LineWidth',2);
set(gca,'FontSize',20)
hold on;
plot(xc,conv(f(xc),-thc,'same'),'.-k','MarkerSize',30,'LineWidth',2);
legend('\beta(1)f+\beta(2)f'' + \beta(3) f''''','conv(f(xc),thc)','location','SouthWest')
title('coarse conv')

%%
subplot(2,2,3)
fig.Name = 'z';
plot(x,z(x),'-r','LineWidth',2);
set(gca,'FontSize',20)
hold on;
plot(xf,conv(f(xf),-thc,'same'),'.-k','MarkerSize',30,'LineWidth',2);
legend('\beta(1)f+\beta(2)f'' + \beta(3) f''''','conv(f(xf),thc)','location','SouthWest')
title('fine conv (coarse stencil)')
%%
subplot(2,2,4)
fig.Name = 'z';
plot(x,z(x),'-r','LineWidth',2);
set(gca,'FontSize',20)
hold on;
plot(xf,conv(f(xf),-thf,'same'),'.-k','MarkerSize',30,'LineWidth',2);
legend('\beta(1)f+\beta(2)f'' + \beta(3) f''''','conv(f(xf),thf)','location','SouthWest')
title('fine conv (prol. stencil)')


