%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% classdef opEye
%
% Matrix-free implementation of identity matrix
% 
classdef opEye 
    properties
        n     % number of columns/rows
        Amv   % mat-vec, Amv(x) = x
        ATmv  % transpose mat-vec, ATmv(x)=x
    end
    properties (Dependent)
      m % number of rows, equal to this.n
   end
    
    methods
        function this = opEye(n)
            % constructor, opEye(n)
            this.n = n;
            this.Amv = @(x) x;
            this.ATmv = @(x) x;
        end
        
        function z = mtimes(this,x)
            z = this.Amv(x);
        end
        
        function this = convertGPUorPrecision(this,useGPU,precision)
            % do nothing
        end

        function m = get.m(this)
            m = this.n;
        end
    end
end

