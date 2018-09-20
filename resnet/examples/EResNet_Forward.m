%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% Example for forward propagation through a ResNet
%

clear
param.act = @tanhActivation;

% Width of each block (including initial conditions)
n = [2;2]*ones(1,1000);

a = -.001;
 for i=1:size(n,2)
    if n(1,i) ~= n(2,i)
        P{i} = opZero(n(2,i),n(1,i));
    else
        P{i} = opEye(n(1,i));
    end
    K{i} = [a  -.2; .2 a];
    b{i} = zeros(n(2,i),1);
 end
N = length(P);
param.P = P;
Y0 = [1;1];
param.h = 1e-1;
param.n = n;

Kb = [cell2vec(K); cell2vec(b)];

%% Run the NN forward
[Y,Yall] = ResNetForward(Kb,Y0,param);
[Y2,Yall2] = ResNetForward(Kb,-Y0,param);
%%
ya = reshape(cell2vec(Yall),2,[]);
ya2 = reshape(cell2vec(Yall2),2,[]);
figure(1);clf
plot(ya(1,1),ya(2,1),'-or','MarkerSize',20)
hold on

plot(ya(1,:),ya(2,:),'-r','LineWidth',3)
set(gca,'FontSize',20)

plot(ya2(1,1),ya2(2,1),'-ob','MarkerSize',20)
hold on

plot(ya2(1,:),ya2(2,:),'-b','LineWidth',3)
set(gca,'FontSize',20)