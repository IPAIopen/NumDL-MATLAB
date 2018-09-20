function [Fc,dF,d2F] = quadObjFun(A,b,xc,S)

res = A(S,:)*xc - b(S);

Fc = 0.5*res'*res;
dF = A(S,:)'*res;
d2F = A(S,:)'*A(S,:);

