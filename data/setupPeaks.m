%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [Y,C] = setupPeaks(np, nc, ns)
%
% setup data for peaks example
%
function[Y,C] = setupPeaks(np, nc, ns)
% generates PEAKs example
if not(exist('np','var')) || isempty(np)
    np = 8000;
end

if not(exist('nc','var')) || isempty(nc)
    nc = 5;
end

if not(exist('ns','var')) || isempty(ns)
    ns = 256;
end


[xx,yy,cc] = peaks(ns);
t1 = linspace(min(xx(:)),max(xx(:)),ns);
t2 = linspace(min(yy(:)),max(yy(:)),ns);

% Binarize it
mxcc = max(cc(:)); mncc = min(cc(:)); 
hc = (mxcc - mncc)/(nc);
ccb = zeros(size(cc));
for i=1:nc
    ii = find( (mncc + (i-1)*hc)< cc & cc <= (mncc+i*hc));
    ccb(ii) = i-1;
end

figure(1); clf;
imagesc(t1,t2,reshape(ccb,ns,ns))
% rng('default');
% rng(2)

% draw same number of points per class
Y = [];
npc = ceil(np/nc);
for k=0:nc-1
   xk = [xx(ccb==k) yy(ccb==k)];
   inds = randi(size(xk,1),npc,1);
   
   Y = [Y; xk(inds,:)];
end

C = kron(eye(nc),ones(npc,1));
Y = Y';
C = C';
