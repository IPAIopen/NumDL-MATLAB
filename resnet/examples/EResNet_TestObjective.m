%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% test derivative of ResNet objective function using Peaks example
%

clear all;
[Y,C] = setupPeaks();

%%
param.act = @smoothRelU;
% Width of each block (including initial conditions)
n = [2  5   5   5   5   10; ...
     5  5   5   5  10   3];

 for i=1:size(n,2)
    if n(1,i) ~= n(2,i)
        P{i} = opZero(n(2,i),n(1,i));
    else
        P{i} = opEye(n(1,i));
    end
    K{i} = randn(n(2,i),n(1,i));
    b{i} = randn(n(2,i),1);
end
N = length(P);
param.P = P;
param.h=1;
param.n = n;
Y0 = randn(2,100);
%% Run the NN forward
Kb = [cell2vec(K); cell2vec(b)];
W  = randn((n(end,end)+1)*size(C,1),1);
x0 = [Kb;W];
[Ec,dE,H] = ResNetObjFun(x0,Y,C,numel(Kb),param);

dx = randn(size(x0));
dEdx = dot(dE,dx);
for k=1:10
    h = 10^(-k);
    Et = ResNetObjFun(x0+h*dx,Y,C,numel(Kb),param);    
    fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',h,abs(Ec-Et),abs(Et-Ec-h*dEdx));    
end
