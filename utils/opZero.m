%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% classdef opZero
%
% Matrix-free implementation of zeros(m,n)
% 
classdef opZero
    properties
        m    % number of rows
        n    % number of columns
        Amv  % mat-vec, Amv(x) = 0;
        ATmv % transpose mat-vec, ATmv(x) = 0;
    end
    
    methods
        function this = opZero(m,n)
            % constructor, opZero(m,n)
            this.m = m;
            this.n = n;
            this.Amv = @(x) 0;
            this.ATmv = @(x) 0;
        end
        function z = mtimes(this,x)
            z = this.Amv(x);
        end
        function this = ctranspose(this)
            temp   = this.m;
            this.m = this.n;
            this.n = temp;
            temp   = this.Amv;
            this.Amv = this.ATmv;
            this.ATmv = temp;
        end
      
        function this = convertGPUorPrecision(this,useGPU,precision)
            % do nothing
        end

    end
end

