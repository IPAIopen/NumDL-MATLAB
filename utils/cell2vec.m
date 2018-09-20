%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% a = cell2vec(A)
%
% vectorizes a cell array
function a = cell2vec(A)

a = [];
for i=1:length(A)
    a = [a ; A{i}(:)];
end