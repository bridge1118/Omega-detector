%
% Third version extract 'ellipse' area as features (if extend==0 => single area)
% multi part (deform part=15)
% !sift feature - 
% !gmm classifier
%
close all;clear all;clc;

try
    vl_version verbose    
catch
    run('~/Documents/MATLAB/Apps/vlfeat-0.9.20/toolbox/vl_setup')
end

addpath( genpath('apps') );
addpath( genpath('imgs') );
addpath( genpath('libs') );
load( 'apps/deformedTemplates');

%
% Part Variables
PART = size(deformedTemplates,2);

%% Training Stage
disp('----------Training Stage----------');

[imsT, imsTNum] = ReadImages('imgs/posAll', 0);
featureT = cell( imsTNum, PART );

fprintf('Pre-processing & Extracting features...');tic
for idx = 1 : imsTNum
    % pre-processing
    binSize = 8 ;
    magnif = 1 ;
    imsT{idx} = vl_imsmooth(imsT{idx}, sqrt((binSize/magnif)^2-.25));
    
    [dsift(idx).keypts, dsift(idx).descrs] = ...
        vl_dsift(imsT{idx}, 'size', binSize);
end
fprintf(['Done! ' '(elapsed time: ' num2str(toc) ' seconds)\n']);
clear binSize magnif

fprintf('Selecting features in ellipse area...');tic
for part = 1 : PART 
    for idx = 1 : imsTNum
        
        dsift(idx).keypts(3,:) = 0; % init
        % keypts(1,:)->cals
        % keypts(2,:)->rows
        [rows, cals] = find( deformedTemplates{idx,part} ); % centers
        
        for k = 1 : length(cals) % pick if keypt is in the ellipse area
            dsift(idx).keypts(3, ...
                find(dsift(idx).keypts(1,:)==cals(k) &...
                     dsift(idx).keypts(2,:)==rows(k)) ...
            ) = 1;
        end
        
        selectedFeature = []; % selected feature per part
        for k = 1 : length(dsift(idx).keypts)
            if dsift(idx).keypts(3,k)==1
                selectedFeature = [selectedFeature, dsift(idx).descrs(:,k)];
            end
        end
        featureT{idx, part} = selectedFeature;
        
    end
end
fprintf(['Done! ' '(elapsed time: ' num2str(toc) ' seconds)\n']);
clear part idx cals rows deformedTemplates k selectedFeature 

%% SVM
fprintf('Parsing SVM instance...');tic
dTrain = cell(imsTNum, 1);
maxLength = 0;
for img = 1 : imsTNum
    for part = 1 : PART    
        dTrain{img} = [ dTrain{img} , featureT{img,part}(:)' ];
    end
    if length(dTrain{img}) > maxLength
        maxLength = length(dTrain{img});
    end
    %cTrain = ones( length(dTrain{part}),1 );
end
instanceP = zeros(imsTNum, maxLength);
labelP = ones(imsTNum,1);
for img = 1 : imsTNum
    currentLength = length(dTrain{img});
    instanceP(img,1:currentLength) = dTrain{img};
end

fprintf(['Done! ' '(elapsed time: ' num2str(toc) ' seconds)\n']);
clear maxLength img part dTrain currentLength

%% PCA
tic;
K = 700;
instTmp = [];
p = 15;
for idx = 1 : p
    disp(['---PCA ' num2str(idx) '/' num2str(p)]);
    eigenVect = princomp( instanceP(:, 1:(end/p)) );
    tmp = instanceP(:,1:(end/p)) * eigenVect(:,1:K);
    instTmp = [ instTmp , tmp ];
    instanceP = instanceP(:, (end/p):end);
end
instanceP = instTmp;
fprintf(['PCA Finished! ' '(elapsed time: ' num2str(toc) ' seconds)\n']);
clear instTmp idx eigenVect tmp

disp('Training GMM...');
FJ_params = { 'Cmax', 25, 'thr', 1e-3, 'animate', 0 };
bayesS = gmmb_create(instanceP, ones(131,1), 'FJ', FJ_params{:});
fprintf(['Done! ' '(elapsed time: ' num2str(toc) ' seconds)\n']);

return
%% Negative Samples
disp('---Negative Samples');
[imsN, imsNNum] = ReadImages('imgs/negAll', 0);

fprintf('Pre-processing & Extracting features...');tic
for idx = 1 : imsNNum
    % pre-processing
    binSize = 8 ;
    magnif = 1 ;
    imsN{idx} = vl_imsmooth(imsN{idx}, sqrt((binSize/magnif)^2-.25));
    
    [dsiftN(idx).keypts, dsiftN(idx).descrs] = ...
        vl_dsift(imsN{idx}, 'size', binSize);
end
fprintf(['Done! ' '(elapsed time: ' num2str(toc) ' seconds)\n']);
clear binSize magnif idx

%% Neg SVM
fprintf('Parsing SVM instance...');tic
for img = 1 : imsNNum
    instanceN(img,:) = dsiftN(img).descrs(:)';
end
tmp = [];
front = 1;
last = size(instanceP, 2);
leng = size(instanceP, 2);
while (1)
    tmp = [ tmp ; instanceN(:,front:last) ];
    front = front + leng;
    last = last + leng;
    if last > size(instanceN, 2)
        instanceN = tmp(1:2*size(instanceP,1),:);
        break
    end
end
clear tmp front last leng img
labelN = zeros(size(instanceN,1),1);
fprintf(['Done! ' '(elapsed time: ' num2str(toc) ' seconds)\n']);

%% Training SVM
fprintf('Training SVM...\n');tic
labels = [ labelP ; labelN ];
instance = double([ instanceP ; instanceN ]);
model = svmtrain(labels, instance, '-c 1 -g 0.07');
fprintf(['Done! ' '(elapsed time: ' num2str(toc) ' seconds)\n']);


%% Testing Stage
disp('----------Testing Stage----------');
[imsV, imsVNum] = ReadImages('imgs/test_bridge', 0);
binSize = 8 ;
magnif = 1 ;
for img = 1 : imsVNum
    %disp(['----Detecting ' num2str(img) ' of ' num2str(imsVNum)]);tic
    
    % pre-processing
    imsV{img} = vl_imsmooth(imsV{img}, sqrt((binSize/magnif)^2-.25));
    
    % extracting feature
    [dsiftV(img).keypts, dsiftV(img).descrs] = ...
        vl_dsift(imsV{img}, 'size', binSize);
    
    
    instanceV(img) = dsiftV(img).descrs(:)';
    instanceV(img) = double(instanceV);
    labelsV(img) = ones(1,1);
    
end

[predict_label, accuracy, dec_values] = ...
        svmpredict(labelsV, instanceV, model);
%disp(['elapsed time: ' num2str(toc) ' seconds']);

clear binSize magnif