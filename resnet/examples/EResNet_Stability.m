%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% demonstrating the importance of stability for a simple ResNet
%

close all; clear all; clc;

model = 'M-matrix';
% activation = @(x) max(x,0);
activation = @(x) tanh(x);

switch model
    case 'M-matrix'
        getK = @(th) -eye(3)*sum(th)-[0 -th(1) -th(2); -th(2) 0 -th(1); -th(1) -th(2) 0];
end

N   = [5 100 200]; % number of time steps for coarse, fine, and true
T   = 10; 
h   = T./N;
nex = 20;
Y0  = .1*randn(3,nex); % training
Y0t = .1*randn(3,nex); % test
nth = 101; % number of cells for objective functions
%% get true labels
K = getK([1;1]);
C = Y0; Ct = Y0t;
for j=1:N(3)
    C  = C  + h(3)*activation(K*C);
    Ct = Ct + h(3)*activation(K*Ct);
end

%% compute misfit with few time steps
th   = linspace(0.2,2,nth);
Phic  = zeros(nth,nth);
Phict = zeros(nth,nth);
for k1=1:nth
    for k2=1:nth
        Cpred = Y0;
        Cpredt = Y0t;
        K = getK([th(k1);th(k2)]);
        for j=1:N(1)
            Cpred  = Cpred + h(1)*activation(K*Cpred);
            Cpredt = Cpredt + h(1)*activation(K*Cpredt);
        end
        Phic(k1,k2)  = 0.5*norm(Cpred-C,'fro')^2;
        Phict(k1,k2) = 0.5*norm(Cpredt-Ct,'fro')^2;
    end
end
%%
figure(1); clf;
subplot(2,3,1);
contour(th,th,Phic,100); 
hold on;
plot(1,1,'.r','MarkerSize',20);
title(sprintf('objective, h=%1.2f',h(1)));
ylabel('few time steps')
subplot(2,3,2);
contour(th,th,Phict,100);
hold on;
plot(1,1,'.r','MarkerSize',20);
title(sprintf('objective (test), h=%1.2f',h(1)));
subplot(2,3,3);
imagesc(th,th,flipud(abs(Phic-Phict)'));
title(sprintf('abs. diff, h=%1.2f',h(1)));
colorbar

%% compute misfit with many time steps
Phif  = zeros(nth,nth);
Phift = zeros(nth,nth);
for k1=1:nth
    for k2=1:nth
        Cpred = Y0;
        Cpredt = Y0t;
        K = getK([th(k1);th(k2)]);
        for j=1:N(2)
            Cpred  = Cpred  + h(2)*activation(K*Cpred);
            Cpredt = Cpredt + h(2)*activation(K*Cpredt);
        end
        Phif(k1,k2)  = 0.5*norm(Cpred-C,'fro')^2;
        Phift(k1,k2) = 0.5*norm(Cpredt-Ct,'fro')^2;
    end
end
%%
subplot(2,3,4);
contour(th,th,Phif,100);
hold on;
plot(1,1,'.r','MarkerSize',20);
title(sprintf('objective, h=%1.2f',h(2)));
ylabel('many time steps')

subplot(2,3,5);
contour(th,th,Phift,100);
hold on;
plot(1,1,'.r','MarkerSize',20);
title(sprintf('objective (test), h=%1.2f',h(2)));
subplot(2,3,6);
imagesc(th,th,flipud(abs(Phif-Phift)'));
title(sprintf('abs. diff, h=%1.2f',h(2)));
colorbar

return
%% for printing figures;
fig = figure(10); clf;
fig.Name = 'Phic';
contour(th,flipud(th),Phic,100,'LineWidth',2); 
hold on;
plot(1,1,'.r','MarkerSize',40);

fig = figure(11); clf;
fig.Name = 'Phict';
contour(th,flipud(th),Phict,100,'LineWidth',2); 
hold on;
plot(1,1,'.r','MarkerSize',40);

fig = figure(12); clf;
fig.Name = 'Phic-Phict';
diffc = abs(Phic-Phict);
imagesc(th,th,(diffc'));
axis xy
cb = colorbar
cb.Ticks = [min(diffc(:)) max(diffc(:))];
axis square
cb.Position =[0.8405 0.1095 0.02 0.7155]

fig = figure(13); clf;
fig.Name = 'Phif';
contour(th,flipud(th),Phif,100,'LineWidth',2); 
hold on;
plot(1,1,'.r','MarkerSize',40);

fig = figure(14); clf;
fig.Name = 'Phift';
contour(th,flipud(th),Phift,100,'LineWidth',2); 
hold on;
plot(1,1,'.r','MarkerSize',40);

fig = figure(15); clf;
fig.Name = 'Phif-Phift';
difff = abs(Phif-Phift);
imagesc(th,th,difff');
axis xy
cb = colorbar
cb.Ticks   = [min(difff(:)) max(difff(:))];
axis square
cb.Position =[0.8405 0.1095 0.02 0.7155]
%%
for k=10:15
    fig = figure(k);
    set(gca,'FontSize',30);
    axis square
    set(gca,'XTick',[min(th) max(th)],'YTick',[ min(th) max(th)])
    printFigure(gcf,[fig.Name '.png'],'printOpts','-dpng','printFormat','.png');
end

