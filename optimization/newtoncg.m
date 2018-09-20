%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [x,his,xAll] = newtoncg(fun,x0,param)
%
% Newton-CG method with Armijo linesearch
%
% Inputs:
%  fun   - objective function, e.g., fun = @(x,varargin) Rosenbrock(x)
%  x0    - starting guess
%  param - parameters for algorithm. Supported fields
%          maxIter   - maximum number of iterations
%          tolCG     - tolerance for PCG solver              (default: 1e-2)
%          maxIterCG - maximum number of CG iterations       (default: 10)
%          out       - flag controlling output               (default: 1)
%
% Outputs:
%  x     - last iterate
%  his   - iteration history
%  xAll  - all iterates

function [x,his,xAll] = newtoncg(fun,x0,param)

if nargin==0
   E = @(x,varargin) Rosenbrock(x);
   W = [4;2];
   param = struct('maxIter',30,'maxStep',10);
   [W,his,xAll] = feval(mfilename,E,W,param);
   fprintf('numerical solution: W = [%1.4f, %1.4f]\n',W);
   return
end

x = x0; xAll = [];
[obj,dobj,H] = fun(x);

% get paramters
if isfield(param,'tolCG'); 
    tolCG = param.tolCG;
else
    tolCG = 1e-2;
end
if isfield(param,'maxIterCG')
    maxIterCG = param.maxIterCG;
else
    maxIterCG = 10;
end

if isfield(param,'out')
    out=param.out; 
else
    out=1; 
end

mu = 1; 
his = zeros(param.maxIter,2);
if out==1
    fprintf('=== %s (maxIter: %d) ===\n',mfilename,param.maxIter);
    fprintf('iter\t obj func\tnorm(grad)\n');
end

for j=1:param.maxIter
    
    his(j,:) = [obj,norm(dobj)];
    
    if nargout>2; xAll = [xAll x]; end;
    
    if out==1; fprintf('%3d.0\t%3.2e\t%3.2e\n',j,his(j,:)); end
    [s,FLAG,RELRES,ITER,RESVEC] = pcg(H,-dobj,tolCG,maxIterCG);

    % resort to steepest descent if pcg fails
    if norm(s)==0
        s = -dobj/norm(dobj);
    end
    
    % test if s is a descent direction
    if s(:)'*dobj(:) > 0
        s = -dobj;
    end
    % Armijo line search
    cnt = 1;
    while 1
        xtry = x + mu*s;
        [objtry,dobj,H] = fun(xtry,param);
        if out==1; fprintf('%3d.%d\t%3.2e\t%3.2e\n',j,cnt,objtry,norm(dobj)); end

        if objtry< obj
            break
        end
        mu = mu/2;
        cnt = cnt+1;
        if cnt > 10
            warning('Line search break');
            return;
        end
    end
    if cnt == 1
        mu = min(mu*1.5,1);
    end
    x   = xtry;
    obj = objtry;
end