close all; clear all; clc;

rng(20)
n  = 500; nf = 50; nc = 10; m  = 40;
Wtrue = randn(nc,m);
Ktrue = randn(m,nf);
btrue = .1;

Y     = randn(nf,n);
Cobs  = exp(Wtrue*singleLayer(Ktrue,btrue,Y));
Cobs  = Cobs./sum(Cobs,1);

%% test single layer NN objective
x0 = randn(numel(Ktrue)+numel(btrue)+numel(Wtrue),1);

[Ec,dE] = singleLayerNNObjFun(x0,Y,Cobs,m);

assert(isscalar(Ec),'objective function should return scalar');
assert(all(size(dE)==size(x0)),'gradient should be a column vector');

% check derivative
dx = randn(size(x0));

[Ec,dE] = singleLayerNNObjFun(x0,Y,Cobs,m);
dEdx = dE'*dx;

% dF = dF + 1e-2*randn(size(dF));
err    = zeros(30,3);
for k=1:size(err,1)
    h = 2^(-k);
    Et = singleLayerNNObjFun(x0+h*dx,Y,Cobs,m);
    
    err(k,:) = [h, norm(Ec-Et), norm(Ec+h*dEdx-Et)];
    
    fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',err(k,:))
end

figure; clf;
loglog(err(:,1),err(:,2),'-b','linewidth',3);
hold on;
loglog(err(:,1), err(:,3),'-r','linewidth',3);
legend('E0','E1');

% run steepest descent
F = @(x) singleLayerNNObjFun(x,Y,Cobs,m);
param = struct('maxIter',2000,'maxStep',1);
xSol = steepestDescent(F,x0,param);

Ksol = reshape(xSol(1:nf*m),m,nf);
bsol = xSol(nf*m+1);
Wsol = reshape(xSol(nf*m+2:end),nc,m);
%
norm(Ktrue-Ksol)/norm(Ktrue)
norm(bsol-btrue)/norm(btrue)
norm(Wtrue-Wsol)/norm(Wtrue)
%

Cpred  = exp(Wsol*singleLayer(Ksol,bsol,Y));
Cpred  = Cpred./sum(Cpred,1);
assert(norm(Cobs-Cpred)/norm(Cobs) < 0.01,'training accuracy too low');