%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
%[x,his,xAll] = steepestDescent(fun,x0,param)
%
% Steepest Descent method with Armijo linesearch
%
% Inputs:
%  fun   - objective function, e.g., fun = @(x,varargin) Rosenbrock(x)
%  x0    - starting guess
%  param - parameters for algorithm. Supported fields
%          maxIter   - maximum number of iterations
%          maxStep   - maximum step size
%          P         - projection onto feasible set       (default: @(x) x)
%          out       - flag controlling output            (default: 1)
%
% Outputs:
%  x     - last iterate
%  his   - iteration history
%  xAll  - all iterates
function [x,his,xAll] = steepestDescent(fun,x,param)

if nargin==0
   fun = @Rosenbrock;
   x = [4;2];
   param = struct('maxIter',10000,'maxStep',1);
   x = feval(mfilename,fun,x,param);
   fprintf('numerical solution: W = [%1.4f, %1.4f]\n',x);
   return
end
xAll      = [];
mu        = param.maxStep; % max step size
maxIter   = param.maxIter; % max number of iterations
if isfield(param,'P')
    P = param.P;
else
    P = @(x) x;
end
if isfield(param,'out')
    out=param.out; 
else
    out=1; 
end
muLS = 1.0;

if out==1
    fprintf('=== %s (maxIter: %d) ===\n',mfilename,maxIter);
    fprintf('iter\t obj func\t\tnorm(grad)\n');
end
his = zeros(maxIter,2);

x = P(x);
for i=1:maxIter
    [Ec,dE] = fun(x);
    his(i,:) = [Ec,norm(dE)];
    if out==1; fprintf('%3d.0\t%3.2e\t%3.2e\n',i,his(i,:)); end
    if norm(dE)>mu, dE = mu*dE/norm(dE); end;
    
    for LSiter=1:10
        xt = P(x-muLS*dE);
        Et = fun(xt);
        if out==1; fprintf('%3d.%d\t%3.2e\n',i,LSiter,Et); end
        if Et<Ec
            break;
        end
        muLS = muLS/2;
    end
    if LSiter==1
        muLS = min(1, muLS*1.5);
    end
    if LSiter==10 && Et >= Ec
        warning('LSB')
        return
    end        
    x = xt;
    if nargout>2; xAll = [xAll x]; end
end

   

