close all;clear all;clc

load 'model.mat'

disp('----------Testing Stage----------');
[imsV, imsVNum] = ReadImages('imgs/test_bridge', 0);
%instanceV = [];
binSize = 8 ;
magnif = 1 ;
for img = 1 : imsVNum
    
    % pre-processing
    imsV{img} = vl_imsmooth(imsV{img}, sqrt((binSize/magnif)^2-.25));
    
    % extracting feature
    [dsiftV(img).keypts, dsiftV(img).descrs] = ...
        vl_dsift(imsV{img}, 'size', binSize);
    instanceV(img,:) = dsiftV(img).descrs(:)';
end

instanceV = double(instanceV);
labelsV = ones(size(instanceV,1),1);

[predict_label, accuracy, dec_values] = ...
        svmpredict(labelsV, instanceV, model);

clear binSize magnif