%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% C = montageArray(A,ncol)
% 
% Inputs:
%
%  A    - 3D/4D array
%  ncol - specify number of columns in montage. default=[]
%
% Outputs:
% 
%  C    - montage
function C = montageArray(A,ncol)



[m1,m2,m3] = size(A);
if not(exist('ncol','var')) || isempty(ncol)
    ncol = ceil(sqrt(m3));
end
nrow = ceil(m3/ncol);

C = zeros(m1*nrow, m2*ncol);
M = zeros(m1*nrow, m2*ncol);

k=0;

for p=1:m1:(nrow*m1)
  for q=1:m2:(ncol*m2)
    k=k+1;
    if k>m3, 
      break
    end
    C(p:(p+m1-1),q:(q+m2-1)) = A(:,:,k);
    M(p:(p+m1-1),q:(q+m2-1)) = 1;
  end
end

if nargout == 0
  h = imagesc(C);
  set(h, 'AlphaData', M);

  washold = ishold;
  hold on;
  
  [P,Q]= ndgrid(1:m1:((nrow+1)*m1),1:m2:((ncol+1)*m2));
  P = P -0.5;
  Q = Q -0.5;
  plot(Q,P,'color',get(gcf,'color'),'linewidth',3);
  plot(Q',P','color',get(gcf,'color'),'linewidth',3);
  plot(Q,P,'k','linewidth',1);
  plot(Q',P','k','linewidth',1);

  if ~washold,
      hold off;
  end;
  
end


  


