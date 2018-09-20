%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% test derivative of ResNet forward propagation using Peaks example
%

clear
param.act = @smoothRelU;

% Width of each block (including initial conditions)
n = [2  5   5   5  5    10; ...
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
Kb = [cell2vec(K); cell2vec(b)];
%% Run the NN forward
[Y,Yall,dA] = ResNetForward(Kb,Y0,param);
dY0 = randn(size(Y0));
dKb = randn(size(Kb));
dY  = dResNetMatVec(dKb,dY0,Kb,Yall,dA,param);

fprintf('=== Derivative test ========================\n')
for k=1:10
    h = 10^(-k);
    
    Y1 = ResNetForward(Kb+h*dKb,Y0+h*dY0,param);

    
    fprintf('%3.2e  %3.2e  %3.2e\n',h,norm(Y1(:)-Y(:)), norm(Y1(:)-Y(:)-h*dY(:)))
end
%%

fprintf('=== Adjoint test ========================\n')
dZ  = dResNetMatVec(dKb,dY0,Kb,Yall,dA,param);

dW = randn(size(dZ));
t1 = dZ(:)'*dW(:);

[dK1,db1,dY1] = dResNetMatVecT(dW,Kb,Yall,dA,param);

tt = [cell2vec(dK1); cell2vec(db1)];

t2  = tt'*dKb + dot(dY1(:),dY0(:)) ;
fprintf('%3.2e  %3.2e\n',t1,t2)
