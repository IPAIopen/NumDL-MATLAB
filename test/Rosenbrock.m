function [f,df,d2f] = Rosenbrock(x)
% [f,df,d2f] = Rosenbrock(x)
%
% Rosenbrock function. Useful to test optimization algorithms

x = reshape(x,2,[]);
f = (1-x(1,:)).^2 + 100*(x(2,:) - (x(1,:)).^2).^2;

if nargout>1 && size(x,2)==1
    df = [2*(x(1)-1) - 400*x(1)*(x(2)-(x(1))^2); ...
        200*(x(2) - (x(1))^2)];
end

if nargout>2 && size(x,2)==1
    n= 2;
    d2f=zeros(n);
    d2f(1,1)=400*(3*x(1)^2-x(2))+2; d2f(1,2)=-400*x(1);
    for j=2:n-1
        d2f(j,j-1)=-400*x(j-1);
        d2f(j,j)=200+400*(3*x(j)^2-x(j+1))+2;
        d2f(j,j+1)=-400*x(j);
    end
    d2f(n,n-1)=-400*x(n-1); d2f(n,n)=200;
end