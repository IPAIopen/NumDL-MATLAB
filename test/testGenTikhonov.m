nf = 10;
nc = 1;
L = randn(20,nf);
param = struct('L',L,'nc',nc,'h',1);
W  = randn(nf*nc,1);
%% check calls of genTikhonov
[Rc,dR,d2R] = genTikhonov(W,param);
assert(numel(Rc)==1,'first output argument of softMax must be a scalar');
assert(all(size(dR)==size(W)),'size of gradient and W must match');

if isnumeric(d2R)
    assert(all(size(d2R)==[numel(W),numel(W)]),'size of Hessian incorrect');    
elseif isa(d2R,'function_handle')
    assert(all(size(d2R(W))==size(W)),'matrix free Hessian must preserve size');
else
    error('d2R must be either matrix or function');
end
    
% check calls of genTikhonov
param.lambda = 2;
[R2] =genTikhonov(W,param);
assert(norm(2*Rc-R2)<1e-10,'scaling incorrect!');


%% check derivatives and Hessian
W0 = randn(size(W));
dW = randn(size(W));
param.h = rand();

[F,dF,d2F] = genTikhonov(W0,param);
dFdW = dF'*dW;
if isnumeric(d2F)
    d2FdW = dW'*d2F*dW;
else
    d2FdW = dW'*d2F(dW);
end
err    = zeros(10,4);
for k=1:size(err,1)
    h = 2^(-k);
    Ft = genTikhonov(W0+h*dW,param);
    
    err(k,:) = [h, norm(F-Ft), norm(F+h*dFdW-Ft), norm(F+h*dFdW+h^2/2*d2FdW-Ft)];
    
    fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\tE1=%1.2e\n',err(k,:))
end

figure; clf;
loglog(err(:,1),err(:,2),'-b','linewidth',3);
hold on;
loglog(err(:,1), err(:,3),'-r','linewidth',3);
hold on;
loglog(err(:,1), err(:,4),'-k','linewidth',3);
legend('E0','E1','E2');