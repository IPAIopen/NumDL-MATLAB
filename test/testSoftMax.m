a = 3;
b = 2;

Y = randn(2,500);
C = a*Y(1,:) + b*Y(2,:) +2;
C(C>0) = 1; C(C<0) = 0;
C  = [C; 1-C];

W  = [eye(2) ones(2,1)];
W  = W(:);

%% check calls of softmax
[F1,dF1] = softMax(W,C,C);

assert(numel(F1)==1,'first output argument of softMax must be a scalar')
assert(all(size(dF1)==size(W)), 'size of gradient and W must match');

[F2,dF2] = softMax(1e4*W,C,C);
assert( not(isinf(F2)) && not(isnan(F2)), 'Likely an overflow in softMax ');
assert(abs(F2)<1e-9,'loss should be around zero');

[F3,dF3] = softMax(W,[1e4*C [1; 0]],[C [1;0]]);
assert(not(isinf(F3)) && not(isnan(F3)),'Likely an underflow in softMax ')

%% check derivatives and Hessian
W0 = randn(size(W));
dW = randn(size(W));

[F,dF,d2F] = softMax(W0,Y,C);
dFdW = dF'*dW;
d2FdW = dW'*d2F(dW);
% dF = dF + 1e-2*randn(size(dF));
err    = zeros(30,4);
for k=1:size(err,1)
    h = 2^(-k);
    Ft = softMax(W0+h*dW,Y,C);
    
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

%% check derivatives and Hessian
Y0 = randn(size(Y));
dY = randn(size(Y));

[F,dF,d2F,dFY,d2FY] = softMax(W,Y0,C);
dFdY = dFY'*dY(:);
d2FdY = dY(:)'*d2FY(dY);
% dF = dF + 1e-2*randn(size(dF));
err    = zeros(30,4);
for k=1:size(err,1)
    h = 2^(-k);
    Ft = softMax(W,Y0+h*dY,C);
    
    err(k,:) = [h, norm(F-Ft), norm(F+h*dFdY-Ft), norm(F+h*dFdY+h^2/2*d2FdY-Ft)];
    
    fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\tE1=%1.2e\n',err(k,:))
end

figure; clf;
loglog(err(:,1),err(:,2),'-b','linewidth',3);
hold on;
loglog(err(:,1), err(:,3),'-r','linewidth',3);
hold on;
loglog(err(:,1), err(:,4),'-k','linewidth',3);
legend('E0','E1','E2');