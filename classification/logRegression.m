%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
%[E,dEW,d2EW,dEY,d2EY] = logRegression(W,Y,C)
% 
% Evaluates cross-entropy loss function for logistic regression
%
% Inputs:
%
%  W  - current weights of classifier
%  Y  - features
%  C  - labels
%
% Outputs:
% 
%  E   - cross-entropy
%  dEW - gradient w.r.t. W
%  d2W - function handle for matvec with Hessian w.r.t. W
%  dEY - gradient w.r.t. Y
%  d2Y - function handle for matvec with Hessian w.r.t. Y

function[E,dEW,d2EW,dEY,d2EY] = logRegression(W,Y,C)

if nargin == 0
   runMinExample;
   return
end
if size(C,1) ~= 1
    error('logRegression can only handle binary classification.')
end
W = reshape(W,1,[]); addBias = false;
if size(W,2)==size(Y,1)+1
    addBias = true;
    Y = [Y; ones(1,size(Y,2))];
end
n = size(Y,2);
nc = size(C,1);

% the linear model
S = W*Y;



% The cross entropy
posInd = (S>0);
negInd = (S<=0);
E = -sum(C(negInd).*S(negInd) - log(1+exp(S(negInd))) ) - ...
                sum(C(posInd).*S(posInd) - log(exp(-S(posInd))+1) - S(posInd));
E = E/n;

if nargout > 1
    dES  = (C- 1./(1+exp(-S)));
    dEW  = -dES*(Y'/n);
    dEW = dEW(:);
end

if nargout>2
    matW = @(v) reshape(v,nc,[]); % reshape vector into same size of W
    vec = @(V) V(:);
    
    d2E  = @(U) U./(2*cosh(S/2)).^2;
    d2EW = @(v) vec(d2E(matW(v)*Y)*Y')/n + 1e-5*v;
end

if addBias
    W = W(:,1:end-1);
end
if nargout > 3
    dEY  = -(W'*dES)/n;
    dEY  = dEY(:);
end

if nargout>4
    matY = @(v) reshape(v,[],n);
    
    d2EY  = @(v) vec(W'*d2E(W*matY(v)))/n + 1e-5*vec(v);
end

end

function runMinExample

vec = @(x) x(:);
nex = 100;
Y = randn(2,nex);
C = Y(1,:) > 0;
b = 0;
W = [1;1];
E = logRegression(W,Y,C);
[E,dE,d2E] = logRegression(W,Y,C);

h = 1;
err = zeros(3,20);
dW = randn(size(W));
for i=1:size(err,2)
    E1 = logRegression(W+h*dW,Y,C);
    t  = abs(E1-E);
    t1 = abs(E1-E-h*dE(:)'*dW(:));
    t2 = abs(E1-E-h*dE(:)'*dW(:) - h^2/2 * dW(:)'*vec(d2E(dW(:))));
    
    fprintf('%3.2e   %3.2e   %3.2e\n',t,t1,t2)
    
    err(i,1) = abs(E1-E);
    err(i,2) = abs(E1-E-h*dE(:)'*dW(:));
    err(i,3) = t2;
    h = h/2;
end
end