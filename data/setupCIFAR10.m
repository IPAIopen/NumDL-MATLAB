%==============================================================================
% This code is part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [Ytrain,Ctrain,Yval,Cval] = setupCIFAR10(nTrain,nVal,option)
%
function[Ytrain,Ctrain,Yval,Cval] = setupCIFAR10(nTrain,nVal,option)

if nargin==0
    runMinimalExample;
    return;
end

if not(exist('nTrain','var')) || isempty(nTrain)
    nTrain = 50000;
end

if not(exist('nVal','var')) || isempty(nVal)
    nVal = ceil(nTrain/5);
end

if not(exist('option','var')) || isempty(option)
    option = 1;
end

if not(exist('data_batch_1.mat','file')) || ... 
        not(exist('data_batch_2.mat','file')) || ...
        not(exist('data_batch_3.mat','file')) || ...
        not(exist('data_batch_4.mat','file')) || ...
        not(exist('data_batch_5.mat','file'))
    
    warning('CIFAR10 data cannot be found in MATLAB path')
    
    dataDir = [fileparts(which('startupNumDLToolbox.m')) filesep 'data'];
    cifarDir = [dataDir filesep 'CIFAR'];
    doDownload = input(sprintf('Do you want to download https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz (around 175 MB) to %s? Y/N [Y]: ',dataDir),'s');
    if isempty(doDownload)  || strcmp(doDownload,'Y')
        if not(exist(dataDir,'dir'))
            mkdir(dataDir);
        end
        imtz = fullfile(dataDir,'cifar-10-matlab.tar.gz');
        if not(exist(imtz,'file'))
            websave(fullfile(dataDir,'cifar-10-matlab.tar.gz'),'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz');
        end
        im  = untar(imtz,dataDir);
        movefile([dataDir filesep 'cifar-10-batches-mat'],cifarDir);
        delete(imtz)
        addpath(cifarDir);
    else
        error('CIFAR10 data not available. Please make sure it is in the current path');
    end
end

% Reading in the data

load data_batch_1.mat
data1   = double(data);
labels1 = labels;

load data_batch_2.mat
data2   = double(data);
labels2 = labels;

load data_batch_3.mat
data3   = double(data);
labels3 = labels;

load data_batch_4.mat
data4   = double(data);
labels4 = labels;

load data_batch_5.mat
data5   = double(data);
labels5 = labels;

data   = [data1; data2; data3; data4; data5];
labels = [labels1; labels2; labels3; labels4; labels5];
nex = size(data,1);



if nTrain<nex
    ptrain = randperm(nex,nTrain);
else
    ptrain = 1:nex;
end

[Ytrain,Ctrain] = sortAndScaleData(data(ptrain,:),labels(ptrain),option);
Ytrain = Ytrain';
Ctrain = Ctrain';
if nargout>2
    load test_batch.mat
    dataTest   = double(data);
    labelsTest = labels;
    nex = size(dataTest,1);
    if nVal<nex
        pval = randperm(nex,nVal);
    else
        pval = 1:nex;
    end
    [Yval,Cval] = sortAndScaleData(dataTest(pval,:),labelsTest(pval),option);
    Yval = Yval';
    Cval = Cval';
end
function runMinimalExample
[Yt,Ct,Yv,Cv] = feval(mfilename,50,10);
figure(1);clf;
subplot(2,1,1);
montageArray(reshape(Yt(1:32*32,:),32,32,[]),10);
axis equal tight
colormap gray
title('training images');


subplot(2,1,2);
montageArray(reshape(Yv(1:32*32,:),32,32,[]),10);
axis equal tight
colormap gray
title('validation images');


function[X,Y] = sortAndScaleData(X,labels,option)
%[X,Y] = sortAndScaleData(X,labels)
%

% Scale X [-0.5 0.5]
%X  = X/max(abs(X(:))) - 0.5;
if nargin == 2, option = 1; end

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng
if option == 1
  X = bsxfun(@minus, X, mean(X,2)) ;
  n = std(X,0,2) ;
  X = bsxfun(@times, X, mean(n) ./ max(n, 40)) ;
end

if option == 2

  W = (X'*X)/size(X,1);
  [V,D] = eig(W);

  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  X = X*(V*diag(en./max(sqrt(d2), 10))*V');
end

if option == 3  
  X = bsxfun(@minus, X, mean(X,2)) ;
  X = X/200;
end


% Organize labels
[~,k] = sort(labels);
labels = labels(k);
X      = X(k,:);

Y = zeros(size(X,1),max(labels)-min(labels)+1);
for i=1:size(X,1)
    Y(i,labels(i)+1) = 1;
end

