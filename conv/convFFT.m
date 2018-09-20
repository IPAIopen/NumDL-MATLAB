%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% classdef convFFT
%
% 2D convolution using FFTs
%
% Transforms feature using affine linear mapping
%
%      Y(theta,Y0) K(theta) * Y0
%
%  where
%
%      K - convolution matrix

classdef convFFT
     
    properties
        nImg  % image size
        sK    % kernel size: [nxfilter,nyfilter,nInputChannels,nOutputChannels]
        Q     % linear transformation applied to stencil elements, default Q = eye
    end
    
    methods
        function this = convFFT(nImg, sK,varargin)
            
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            nImg = nImg(1:2);
            Q = opEye(prod(sK));
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([ varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            this.nImg = nImg;
            this.sK   = sK;
            this.Q = Q;
        end
        
        function A = getOp(this,K)
            % constructs operator for current weight, K
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
            
            I  = rand(nImgIn(kernel)); I(4:12,4:12,:) = 2;
            Ik = Amv(kernel,theta,I);
            Ik2 = ATmv(kernel,theta,Ik);
            figure(1); clf;
            subplot(1,2,1);
            imagesc(I);
            title('input');
            
            subplot(1,2,2);
            imagesc(Ik(:,:,1));
            title('output');
        end
        
        function [Z,tmp] = Amv(this,theta,Y)
            tmp   = []; % no need to store any intermediates
            nex   = size(Y,4);
            
            Z    = zeros([this.nImg this.sK(4) nex],'like',Y);
            S     = reshape(fft2(getK1(this,theta)),[this.nImg this.sK(3:4)]);
            Yh    = fft2(Y);
            for k=1:this.sK(4)
                T  = S(:,:,:,k) .* Yh;
                Z(:,:,k,:)  = sum(T,3);
            end
            Z = real(ifft2(Z));
        end
        
        function dY = Jthetamv(this,dtheta,~,Y,~)
            dY = getOp(this,this.Q*dtheta(:))*Y;
        end
        
        
        function dtheta = JthetaTmv(this,Z,~,Y,~)
            %  derivative of Z*(A(theta)*Y) w.r.t. theta
            nex    =  size(Y,4);
            
            dth1 = zeros(prod(this.sK(1:3)),this.sK(4),'like',Y);
            Yh   = permute(fft2(Y),[1 2 4 3]);
            Zh   = ifft2(reshape(Z,[this.nImg this.sK(4) nex]));
            
            % get q vector for a given row in the block matrix
            v   = vec(1:prod(this.sK(1:3)));
            q   = getK1(this,v);
            
            I    = find(q(:));
            for k=1:this.sK(4)
                Zk = squeeze(Zh(:,:,k,:));
                tt = squeeze(sum(Zk.*Yh,3));
                tt = real(fft2(tt));
                dth1(q(I),k) = tt(I);
            end
            dtheta = dth1(:);
        end
        
        function Y = ATmv(this,theta,Z)
            
            nex =  size(Z,4);
            Y   = zeros([this.nImg this.sK(3) nex],'like',Z);
            S   = reshape(fft2(getK1(this,theta)),[this.nImg this.sK(3:4)]);
            
            Zh = ifft2(Z);
            for k=1:this.sK(3)
                Sk = squeeze(S(:,:,k,:));
                Y(:,:,k,:) = sum(Sk.*Zh,3);
            end
            Y = real(fft2(Y));
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
        
        function K1 = getK1(this,theta)
            % compute first row of convolution matrix
            theta = reshape(theta,this.sK(1),this.sK(2),[]);
            center = (this.sK(1:2)+1)/2;
            
            K1  = zeros([this.nImg size(theta,3)],'like',theta);
            K1(1:this.sK(1),1:this.sK(2),:) = theta;
            K1  = circshift(K1,1-center);
        end
        function n = nImgOut(this)
            n = [this.nImg(1:2) this.sK(4)];
        end
        
        function theta = initTheta(this)
            sd= 0.1;
            theta = sd*randn(this.sK);
            id1 = find(theta>2*sd);
            theta(id1(:)) = randn(numel(id1),1);
            
            id2 = find(theta< -2*sd);
            theta(id2(:)) = randn(numel(id2),1);
            
            theta = max(min(2*sd, theta),-2*sd);
            theta  = theta - mean(theta)
            
        end
        
    end
end

