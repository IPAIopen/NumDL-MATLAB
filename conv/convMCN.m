%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% classdef convMCN
%
% 2D convolution using MatConvNet
%
% Transforms feature using affine linear mapping
%
%      Y(theta,Y0) K(theta) * Y0
%
% !! needs compiled binaries from MatConvNet; see http://www.vlfeat.org/matconvnet/ !!
%
%  where
%
%      K - convolution matrix
classdef convMCN 
    
    
    properties
        nImg  % image size
        sK    % kernel size: [nxfilter,nyfilter,nInputChannels,nOutputChannels]
        Q
        stride
        pad
    end
    
    methods
        function this = convMCN(nImg, sK,varargin)
            
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            nImg = nImg(1:2);
            stride = 1;
            Q = opEye(prod(sK));
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([ varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            this.nImg = nImg;
            this.sK   = sK;
            this.stride = stride;
            this.Q = Q;
            this.pad    = floor((this.sK(1)-1)/2);
        end
        
        function A = getOp(this,K)
            n   = nFeatIn(this);
            m   = nFeatOut(this);
            Af  = @(Y) this.Amv(K,Y);
            ATf = @(Y) this.ATmv(K,Y);
            A   = LinearOperator(m,n,Af,ATf);
        end


        function runMinimalExample(~)
            nImg   = [16 18];
            sK     = [3 3,1,2];
            kernel = feval(mfilename,nImg,sK,'stride',2);
            theta = rand(sK); 
            theta(:,1,:) = -1; theta(:,3,:) = 1;
            
            I  = rand(nImgIn(kernel)); I(4:12,4:12) = 2;
            Ik = Amv(kernel,theta,I);
            Ik2 = ATmv(kernel,theta,Ik);
            Ik = reshape(Ik,kernel.nImgOut());
            figure(1); clf;
            subplot(1,2,1);
            imagesc(I);
            title('input');
            
            subplot(1,2,2);
            imagesc(Ik(:,:,1));
            title('output');
        end
        
        function [Y,tmp] = Amv(this,theta,Y)
            tmp   = []; % no need to store any intermediates
            
            K   = reshape(this.Q*theta(:),this.sK);
            Y   = vl_nnconv(Y,K,[],'pad',this.pad,'stride',this.stride);
        end

        function dY = Jthetamv(this,dtheta,~,Y,~)
            dY = getOp(this,this.Q*dtheta(:))*Y;
        end
        
        
        function dtheta = JthetaTmv(this,Z,~,Y,~)
            %  derivative of Z*(A(theta)*Y) w.r.t. theta
            % get derivative w.r.t. convolution kernels
            [~,dtheta] = vl_nnconv(Y,zeros(this.sK,'like',Y), [],Z,'pad',this.pad,'stride',this.stride);
            dtheta = this.Q'*dtheta(:);
        end

       function dY = ATmv(this,theta,Z)
            
            theta = reshape(this.Q*theta(:),this.sK);

            crop = this.pad;
            if this.stride==2 && this.sK(1)==3
                crop=this.pad*[1,0,1,0];
            elseif this.stride==2 && this.sK(1)==2
                crop=0*crop;
            end
            dY = vl_nnconvt(Z,theta,[],'crop',crop,'upsample',this.stride);
            if this.stride==2 && this.sK(1)==1
                dY = padarray(dY,[1 1],0,'post');
            end
       end
       function n = nFeatIn(this)
            n = prod(nImgIn(this));
        end
        function n = nFeatOut(this)
            n = prod(nImgOut(this));
        end
        
        function n = nImgIn(this)
           n = [this.nImg(1:2) this.sK(3)];
        end
        
        function n = nImgOut(this)
           n = [this.nImg(1:2)./this.stride this.sK(4)];
        end
        
        function theta = initTheta(this)
            sd= 0.1;
            theta = sd*randn(this.sK);
            id1 = find(theta>2*sd);
            theta(id1(:)) = randn(numel(id1),1);
            
            id2 = find(theta< -2*sd);
            theta(id2(:)) = randn(numel(id2),1);
            
            theta = max(min(2*sd, theta),-2*sd);
            theta  = theta - mean(theta);
            
        end
        
    end
end

