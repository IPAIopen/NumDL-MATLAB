%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
%[w,rho,eta,W] = cgls(Y,c,tol,maxIter,w,out)
%
% conjugate gradient method for solving the least-squares problem
%
% min_w 0.5 |Y*w - c|^2 
%     
% Input:
%  Y       - matrix, e.g., features
%  c       - right hand side, e.g., labels
%  tol     - tolerance,                        (default: 1e-2)
%  maxIter - maximum number of iterations,     (default: min(size(Y,2),10))
%  w       - starting guess                    (default: zeros(size(Y,2)))
%  out     - flag for controling output        (default: 1--> print iter)
%
% Output:
%  w       - last iterate
%  rho     - vector of relative residuals 
%  eta     - vector of norms of current iterates
%  W       - history of all iterates
function [w,rho,eta,W] = cgls(Y,c,tol,maxIter,w,out)


if nargin==0
    runMinimalExample
    return
end

n = size(Y,2);

% set optional parameter
if not(exist('tol','var')) || isempty(tol);          tol = 1e-2; end
if not(exist('maxIter','var')) || isempty(maxIter);  maxIter = min(size(Y,2),10); end
if not(exist('w','var')) || isempty(w);              w = zeros(n,1); end
if not(exist('out','var')) || isempty(out);          out=1; end

if nargout>3, W = w; end
r = c-Y*w;
d = Y'*r;   
normr2 = d'*d;
rho = zeros(maxIter,1); eta = zeros(maxIter,1);

if out
    fprintf('=== %s (tol: %1.1e, maxIter: %d) ===\n',mfilename,tol,maxIter); 
    fprintf('iter\trelres\tnorm(w)\n'); 
end

for j=1:maxIter
    Ad = Y*d; 
    alpha = normr2/(Ad'*Ad);
    w  = w + alpha*d;
    r  = r - alpha*Ad;
    s  = Y'*r;
    normr2New = s'*s;
    if normr2New<tol
        return
    end
    beta = normr2New/normr2;
    normr2 = normr2New;
    d = s + beta*d;
    rho(j) = norm(r)/norm(c);
    eta(j)  = norm(w);
    if nargout>3, W = [W w]; end
    if out, fprintf('%3d\t%3.1e\t%3.1e\n',j,norm(r)/norm(c),norm(w)); end
end

function runMinimalExample
Y     = randn(10,2); 
wtrue = [1;1];
c     = Y*wtrue;

[w,rho,eta,W] = feval(mfilename,Y,c);
fprintf('true w = [%1.2f,%1.2f]\n',wtrue);
fprintf('est. w = [%1.2f,%1.2f]\n',w);


