function [images, imNumb] = ReadImages( path, normalize )

fprintf('Reading the images...');tic

%
% Variables
imName = dir( [path '/*.jpg'] );
imNumb  = size(imName, 1);

%
% Output
images = cell(imNumb, 1);

%
% Read images and assign values to struct
for idx = 1 : imNumb
    
    images{idx} = imread([ path '/' imName(idx).name ]);
    
    if size(images{idx}, 3) == 3 % RGB to gray
        images{idx} = rgb2gray( images{idx} );
    end
    images{idx} = imresize(images{idx}, [100 100]);
    images{idx} = single( images{idx} );
    
    if ( normalize ~= 0 ) % Normalize
        images{idx} = images{idx} / 255;
    end
    
end

fprintf('Done!');
fprintf(['(elapsed time: ' num2str(toc) ' seconds)\n']);