%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
%[xc,his,xAll] = sgd(fctn,xc,param)
% 
% Simple implementation of a Stochastic Gradient Descent method with
% momentum.
%
% Inputs: 
%  fctn  - objective function (accepts two arguments: the current iterate
%                        and indices of data points to use in current step)
%  xc    - starting guess
%  param - struct, algorithmic paramter. Supported parameters are
%          lr        - vector, learning rate for each epoch           
%          n         - number of data points overall
%          batchSize - number of examples per batch
%          momentum  - momentum parameter
%          out       - flag controlling output             (default: out=1)
%
% Outputs:
%  xc    - last iterate
%  his   - iteration history
%  xAll  - iterates after each epoch
function [xc,his,xAll] = sgd(fctn,xc,param)


if nargin==0
    A = hilb(10); A = A(:,1:2);
    x = ones(2,1);
    b = A*x;
    
    fctn = @(xc,S) quadObjFun(A,b,xc,S);
    param.lr = 1e-1*ones(100,1);
    param.n  = numel(b);
    param.batchSize = 1;
    param.momentum=0.9;
    xc = 0*x;
    [xOpt,xAll,his] = feval(mfilename,fctn,xc,param);
    xOpt
     
return
end

xAll = [];
% read parameters
lr        = param.lr;
n         = param.n;
batchSize = param.batchSize;
momentum  = param.momentum;
if isfield(param,'out')
    out = param.out;
else
    out = 1;
end

nb = n/batchSize; % number of batches

dF = 0*xc;
his = zeros(numel(lr),2);

if out
    fprintf('=== %s (epochs: %d, batchSize: %d, momentum: %1.2e) ===\n',...
        mfilename, numel(lr),batchSize,momentum);
    fprintf('epoch\tobj fun\t\tnorm step\n');
end


for epoch=1:numel(lr)
    % re-shuffle
    xOld = xc;
    S = reshape(randperm(n),[],nb);
    for batch=1:nb
        Sk = S(:,batch);
        [Fk,dFk] = fctn(xc,Sk); 
        dF = momentum*dF + lr(epoch)*dFk;
        xc = xc - dF;
    end
    %evaluate full objective
    [Fc] = fctn(xc,1:min(n,5000));
    his(epoch,:) = [Fc norm(xc-xOld)];
    fprintf('%3d\t%1.2e\t%1.2e\n',epoch,his(epoch,:))
    
    if nargout>2; xAll = [xAll xc]; end;

end