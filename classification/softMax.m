%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
%[E,dEW,d2EW,dEY,d2EY] = softMax(W,Y,C)
% 
% Evaluates cross-entropy loss function for multinomial classification
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

function[E,dEW,d2EW,dEY,d2EY] = softMax(W,Y,C)


if nargin == 0
   runMinExample;
   return
end
W = reshape(W,size(C,1),[]); addBias = false;
if size(W,2)==size(Y,1)+1
    addBias = true;
    Y = [Y; ones(1,size(Y,2))];
end
n = size(Y,2);
nc = size(C,1);

% the linear model
S = W*Y;

% make sure that the largest number in every row is 0
s = max(S,[],1);
S = S-s;


% The cross entropy
expS = exp(S);
sS   = sum(expS,1);

E = -C(:)'*S(:) +  sum(log(sS)); 
E = E/n;

if nargout > 1
    dES  = -C + expS .* 1./sS;
    dEW  = dES*(Y'/n);
    dEW = dEW(:);
end

if nargout>2
    matW = @(v) reshape(v,nc,[]); % reshape vector into same size of W
    vec = @(V) V(:);
    
    d2E  = @(U) (U.*expS)./sS - expS.*(sum(expS.*U,1)./sS.^2);
    d2EW = @(v) vec(d2E(matW(v)*Y)*Y')/n + 1e-5*v;
end

if addBias
    W = W(:,1:end-1);
end
if nargout > 3
    dEY  = (W'*dES)/n;
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
Y = hilb(500)*255;
Y = Y(:,1:nex);
C = ones(3,nex);
C = C./sum(C,1);
b = 0;
W = hilb(501);
W = W(1:3,:);
E = softMax(W,Y,C);
[E,dE,d2E] = softMax(W,Y,C);

h = 1;
rho = zeros(3,20);
dW = randn(size(W));
for i=1:20
    E1 = softMax(W+h*dW,Y,C);
    t  = abs(E1-E);
    t1 = abs(E1-E-h*dE(:)'*dW(:));
    t2 = abs(E1-E-h*dE(:)'*dW(:) - h^2/2 * dW(:)'*vec(d2E(dW(:))));
    
    fprintf('%3.2e   %3.2e   %3.2e\n',t,t1,t2)
    
    rho(i,1) = abs(E1-E);
    rho(i,2) = abs(E1-E-h*dE(:)'*dW(:));
    rho(i,3) = t2;
    h = h/2;
end
end