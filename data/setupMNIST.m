%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [Ytrain,Ctrain,Yval,Cval] = setupMNIST(nTrain,nVal)
%
% setup and load MNIST data
%
function [Ytrain,Ctrain,Yval,Cval] = setupMNIST(nTrain,nVal)

if nargin==0
    runMinimalExample;
    return;
end

if not(exist('nTrain','var')) || isempty(nTrain)
    nTrain = 50000;
end
if not(exist('nVal','var')) || isempty(nVal)
    nVal = round(nTrain/5);
end

if not(exist('train-images.idx3-ubyte','file')) ||...
        not(exist('train-labels.idx1-ubyte','file'))
    
    warning('MNIST data cannot be found in MATLAB path')
    
    dataDir = [fileparts(which('startupNumDLToolbox.m')) filesep 'data' filesep 'MNIST'];
    
    doDownload = input(sprintf('Do you want to download http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz (around 10 MB) to %s? Y/N [Y]: ',dataDir),'s');
    if isempty(doDownload)  || strcmp(doDownload,'Y')
        if not(exist(dataDir,'dir'))
            mkdir(dataDir);
        end
        imgz = websave(fullfile(dataDir,'train-images.idx3-ubyte.gz'),'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz');
        im = gunzip(imgz);
        delete(imgz)
        
        imgz = websave(fullfile(dataDir,'train-labels.idx1-ubyte.gz'),'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz');
        im = gunzip(imgz);
        delete(imgz)
        addpath(dataDir);
    else
        error('MNNIST data not available. Please make sure it is in the current path');
    end
end

I      = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

% get class probability matrix
C      = zeros(10,numel(labels));
ind    = sub2ind(size(C),labels+1,(1:numel(labels))');
C(ind) = 1;

idx = randperm(size(C,2));

idTrain = idx(1:nTrain);
idVal   = idx(nTrain+(1:nVal));

% Scale images between [-0.5 0.5]
Ytrain = I(:,idTrain);
Ctrain = C(:,idTrain);
Ytrain = Ytrain/max(abs(Ytrain(:))) - 0.5;
[~,k] = sort((1:10)*Ctrain);
Ytrain = Ytrain(:,k);
Ctrain = Ctrain(:,k);

if nargout>2
    Yval = I(:,idVal);
    Cval = C(:,idVal);
    Yval = Yval/max(abs(Yval(:))) - 0.5;
    [~,k] = sort((1:10)*Cval);
    Yval = Yval(:,k);
    Cval = Cval(:,k);
end

function runMinimalExample
[Yt,Ct,Yv,Cv] = feval(mfilename,50,10);
figure(1);clf;
subplot(2,1,1);
montageArray(reshape(Yt,28,28,[]),10);
axis equal tight
colormap(flipud(colormap('gray')))
title('training images');


subplot(2,1,2);
montageArray(reshape(Yv,28,28,[]),10);
axis equal tight
colormap(flipud(colormap('gray')))
title('validation images');





