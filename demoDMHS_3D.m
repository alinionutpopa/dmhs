function [pose3D] = demoDMHS_3D(img_path, output_name, multiplier_3D)

% demoDMHS_3D - apply Deep Multitask Human Sensing (DMHS) CNN model 
%               on a single image on the 3D pose estimation task
% [pose3D] = demoDMHS_3D(img_path, output_name, multiplier_3D, multiplier_sem)
%
% OUTPUT :
% pose3D           - cell array with 6 elements (one for each stage of the
%                    DMHS) each containing the corresponding 3D pose
%                    prediction (17x3)
%
% INPUT  :
% img_path         - path to image used for testing
%                  - it should contain a single person inside a bounding box
% output_name      - output name for mat file with results corresponding to
%                    image to img_path
%                  - the mat file will be saved by default in ./data/results/
% multiplier_3D    - image scales used by the 3D pose estimation task of
%                    DMHS network

if(nargin == 0)
    img_path = './data/images/im1020.jpg';
%     img_path = './data/images/im1037.jpg';
%     img_path = './data/images/im1054.jpg';
%     img_path = './data/images/im2673.png';
%     img_path = './data/images/im2788.png';
end
if(nargin < 2)
    output_name = 'results_im1020';
%     output_name = 'results_im1037';
%     output_name = 'results_im1054';
%     output_name = 'results_im2673';
%     output_name = 'results_im2788';
end
if(nargin < 3)
    multiplier_3D = 0.7:0.1:1;
end;

img = imread(img_path);
% Recommended size for the X axis of img in order to obtain the best
% network results with multiplier_3D (0.7:0.1:1). For different image 
% sizes, please revalidate multiplier_3D.
avgDimX = 386;
resizeFactor = avgDimX / size(img, 1);
img = imresize(img, resizeFactor);

% switch displayModel to 1 for visualizing the results
displayMode = 1;

addpath('./code/');

% use_gpu = 1 for GPU usage (please note that for all 6 stages it requires
%                            11 GB of GPU memory)
% use_gpu = 0 for CPU usage
use_gpu = 1;
% default model_id for DMHS
model_id = 1;
param = config(use_gpu, 0, model_id);
model = param.model(param.modelID);
net = caffe.Net(model.deployFile, model.caffemodel, 'test');

rectangle = [1 1 size(img, 2) size(img, 1)];
% First output parameter of applyModelPreloadedNet_DMHS returns the cell
% arrays containing the processed outputs of each of the network's stages.
% Second output parameter of applyModelPreloadedNet_DMHS returns the raw
% output of the network.
pose3D = applyModelPreloadedNet_DMHS_3D(net, img, param, rectangle, multiplier_3D);
if (~exist('./data/results/', 'dir'))
    mkdir('./data/results/');
end;
save(['./data/results/' output_name '.mat'], 'pose3D', 'img');

if (displayMode == 1)
    figure;
    % image used for testing
    imshow(img);
    title('Test Image');
    figure;
    % Estimated 3D poses corresponding to each stage (1 - 6)
    set(gcf, 'Position', get(0,'Screensize'));
    for i = 1 : 6
        subplot(2, 3, i);
        plotSkel3D(pose3D{i}, 'r');
        title(sprintf('Pose 3D prediction - stage %d', i));
    end
end;


