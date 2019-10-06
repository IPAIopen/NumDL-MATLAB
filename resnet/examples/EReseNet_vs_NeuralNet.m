%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% Compare objective function and gradient for ResNet and Neural Net
%
clear
rng(12)
param.act = @tanhActivation;

% Width of each block (including initial conditions)
n = [5  5 5 5 5  5; ...
     5  5 5 5 5 5];
  n=n/5*4
  n(end)=1

for i=1:size(n,2)
    P{i} = opZero(n(2,i),n(1,i));
    K{i} = .2*randn(n(2,i),n(1,i));
    Kt{i} = .2*randn(n(2,i),n(1,i));
    if n(1,i) == n(2,i)
%         K{i} = K{i}-K{i}';
%         Kt{i} = Kt{i}-Kt{i}';
    end
    b{i} = 0*randn(n(2,i),1);
end
N = length(P);
param.P = P;
param.h=1;
param.n = n;
Y0 = randn(n(1),100);
Kb = [cell2vec(K); cell2vec(b)];
Kbt = [cell2vec(Kt); cell2vec(b)];
%% Run the NN forward
[Y,Yall,dA] = ResNetForward(Kb,Y0,param);


C = ResNetForward(Kbt,Y0,param);

res = Y-C;
Jc = norm(Y-C,'fro')/norm(C,'fro');
[dJdK,dJdb,dJdY] = dResNetMatVecT(res,Kb,Yall,dA,param);

dJ =  [cell2vec(dJdK); ];
fig = figure(1);clf
fig.Name = 'dJ-NN'
% subplot(1,2,1)
histogram(dJ/norm(C,'fro'),100)

%%
fig = figure(2);clf
fig.Name = 'obj-NN'
h = linspace(-.1,1,500);
f1 = 0*h;
for k=1:numel(h)
    [Yt,Yall,dA] = ResNetForward(h(k)*Kb+(1-h(k))*Kbt,Y0,param);
    rest = Yt-C;
    f1(k) = norm(rest,'fro')/norm(C,'fro');
end
plot(h,f1,'linewidth',3)
xlabel('t')

%%
for i=1:size(n,2)
    if n(1,i) ~= n(2,i)
        P{i} = opZero(n(2,i),n(1,i));
    else
        P{i} = opEye(n(1,i));
    end
end
param.P = P;

[Y2,Yall2,dA2] = ResNetForward(Kb,Y0,param);

C = ResNetForward(Kbt,Y0,param);

res2 = Y2-C;
Jc2 = norm(res2,'fro')/norm(C,'fro');
[dJdK,dJdb,dJdY] = dResNetMatVecT(res2,Kb,Yall2,dA2,param);

fig = figure(3);clf
fig.Name = 'dJ-Res'
dJ2 =  [cell2vec(dJdK);];
histogram(dJ2/norm(C,'fro'),100)
%%
fig = figure(4);clf
fig.Name = 'obj-Res'
h = linspace(-.1,1,500);
f2 = 0*h;
for k=1:numel(h)
    [Yt,Yall,dA] = ResNetForward(h(k)*Kb+(1-h(k))*Kbt,Y0,param);
    rest = Yt-C;
    f2(k) = norm(rest,'fro')/norm(C,'fro');
end
plot(h,f2,'linewidth',3)
xlabel('t')

%%
figDir = '/Users/lruthot/Dropbox/Projects/NumDL-CourseNotes/images/'
for k=1:4
    fig=figure(k);
    set(gca,'FontSize',30)
    axis square 
    xlabel([])
    printFigure(gcf,fullfile(figDir,['NNvsResNN_' fig.Name '.png']))

end
    
