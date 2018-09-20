close all; clear all; clc;

n  = 10;
nf = 7;
m  = 4;

K  = randn(m,nf);
Y  = randn(nf,n);
b  = randn();


%% test single layer
Z = singleLayer(K,b,Y);

assert(all(size(Z)==[m,n]),'output sizes must be m x n');

%% derivative check for K
dK = randn(m,nf);
b = 0;
[Z,JKt,~,~,JK] = singleLayer(K,b,Y);
dZ = JK(dK);

err    = zeros(30,3);
for k=1:size(err,1)
    h = 2^(-k);
    Zt = singleLayer(K+h*dK,b,Y);
    
    err(k,:) = [h, norm(Z-Zt,'fro'), norm(Z+h*dZ-Zt,'fro')];
    
    fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',err(k,:))
end

% test adjoint
V = randn(m,nf);
W = randn(m,n);
t1 = sum(sum(W.*JK(V)));
t2 = sum(sum(V.*JKt(W)));


fig = figure; clf;
fig.Name = 'checkDerivative for K';
loglog(err(:,1),err(:,2),'-b','linewidth',3);
hold on;
loglog(err(:,1), err(:,3),'-r','linewidth',3);
legend('E0','E1');
adjErr = norm(t1-t2)/norm(t1);
title(sprintf('adjoint error: %1.2e',adjErr))

if adjErr>1e-12
    error('check adjoint for K')
end

%% derivative check for b
b  = randn();
db = randn();
[Z,~,Jbt,~,~,Jb] = singleLayer(K,b,Y);
dZ = Jb(db);

err    = zeros(30,3);
for k=1:size(err,1)
    h = 2^(-k);
    Zt = singleLayer(K,b+h*db,Y);
    
    err(k,:) = [h, norm(Z-Zt,'fro'), norm(Z+h*dZ-Zt,'fro')];
    
    fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',err(k,:))
end

% test adjoint
V = randn();
W = randn(m,n);
t1 = sum(sum(W.*Jb(V)));
t2 = sum(sum(V.*Jbt(W)));


fig = figure; clf;
fig.Name = 'checkDerivative for b';
loglog(err(:,1),err(:,2),'-b','linewidth',3);
hold on;
loglog(err(:,1), err(:,3),'-r','linewidth',3);
legend('E0','E1');
adjErr = norm(t1-t2)/norm(t1);
title(sprintf('adjoint error: %1.2e',adjErr))

if adjErr>1e-12
    error('check adjoint for b')
end

%% derivative check for Y
Y  = randn(nf,n);
dY = randn(nf,n);
[Z,~,~,JYt,~,~,JY] = singleLayer(K,b,Y);
dZ = JY(dY);

err    = zeros(30,3);
for k=1:size(err,1)
    h = 2^(-k);
    Zt = singleLayer(K,b,Y+h*dY);
    
    err(k,:) = [h, norm(Z-Zt,'fro'), norm(Z+h*dZ-Zt,'fro')];
    
    fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',err(k,:))
end

% test adjoint
V = randn(nf,n);
W = randn(m,n);
t1 = sum(sum(W.*JY(V)));
t2 = sum(sum(V.*JYt(W)));


fig = figure; clf;
fig.Name = 'checkDerivative for Y';
loglog(err(:,1),err(:,2),'-b','linewidth',3);
hold on;
loglog(err(:,1), err(:,3),'-r','linewidth',3);
legend('E0','E1');

adjErr = norm(t1-t2)/norm(t1);
title(sprintf('adjoint error: %1.2e',adjErr))

if adjErr>1e-12
    error('check adjoint for Y')
end
