%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [L] = getLaplacian(nImg,h)
%
% generates a discrete Laplacian
%
% Inputs:
%   nImg  - number of pixels in each dimension
%   h     - pixel size
%
% Output:
%   L     - discrete Laplacian, sparse matrix

function[L] = getLaplacian(nImg,h)


if nargin==0
    runMinimalExample;
    return
end

dim = ndims(nImg);

d2dx = @(n,h) 1/h^2*spdiags(ones(n,1)*[1  -2  1],-1:1,n,n);

switch dim
    case 1
        d2dx = d2dx(nImg,h);
    case 2
        d2dx1 = d2dx(nImg(1),h(1));
        d2dx2 = d2dx(nImg(2),h(2));
        
        L = kron(speye(nImg(2)),d2dx1) + kron(d2dx2,speye(nImg(1)));
    case 3
        d2dx1 = d2dx(nImg(1),h(1));
        d2dx2 = d2dx(nImg(2),h(2));
        d2dx3 = d2dx(nImg(3),h(3));
        
        L = kron(speye(nImg(3)),kron(speye(nImg(2)),d2dx1)) +...
            kron(speye(nImg(3)),kron(d2dx2,speye(nImg(1)))) +...
            kron(d2dx3         ,kron(speye(nImg(2)),speye(nImg(1))));

end

function runMinimalExample
n = [32 32];
h = [1 1]./n;
x = h/2:h:1;
[X,Y] = ndgrid(x);

u    = cos(pi*X(:).*Y(:));
Lapu = -pi^2*(X(:).^2+Y(:).^2).*u;

Lap = feval(mfilename,n,h);
Laput = Lap*u;

figure(1); clf;
subplot(1,3,1);
imagesc(reshape(Lapu,n));
cax = caxis;
axis equal tight
title('Lap*u, true')

subplot(1,3,2);
imagesc(reshape(Laput,n));
caxis(cax);
axis equal tight
title('Lap*u, approx')

subplot(1,3,3);
imagesc(reshape(Laput-Lapu,n));
axis equal tight
title('error')

for k=1:3; subplot(1,3,k); set(gca,'FontSize',20); end;