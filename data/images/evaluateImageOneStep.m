function [pose3D, pose2D, bodyPartLabeling] = evaluateImageOneStep(img, output_name, multiplier_3D, multiplier_sem)

if(nargin == 0)
%     img = imread('./data/images/im1012.jpg');
    img = imread('./data/images/im1020.jpg');
%     img = imread('./data/images/im1054.jpg');
end
if(nargin < 2)
%     output_name = './data/results/results_im1012.mat';
    output_name = './data/results/results_im1020.mat';
%     output_name = './data/results/results_im1054.mat';
end
if(nargin < 3)
    multiplier_sem = 0.4:0.1:1; 
    multiplier_3D = 0.7:0.1:1; 
end

avgDimX = 394.76;
resizeFactor = avgDimX / size(img, 2);
img = imresize(img, resizeFactor);

% switch displayModel to 1 for visualizing the results 
displayMode = 1;

addpath('./code/');
addpath(genpath('./code/util/'));

% use_gpu = 1 for GPU usage
% use_gpu = 0 for CPU usage
use_gpu = 1;
model_id = 1;
param = config(use_gpu, 0, model_id);
model = param.model(param.modelID);
net = caffe.Net(model.deployFile, model.caffemodel, 'test');

rectangle = [1 1 size(img, 2) size(img, 1)];
[predictions, network_output] = applyModelPreloadedNet_DMHS(net, img, param, rectangle, multiplier_3D, multiplier_sem);
pose2D = predictions{1};
bodyPartLabeling = predictions{2};
pose3D = predictions{3};
save(output_name, 'pose3D', 'pose2D', 'bodyPartLabeling', 'img', 'network_output');

if (displayMode == 1)
    figure; 
    imshow(img);
    title('Test Image');
    figure;
    set(gcf, 'Position', get(0,'Screensize'));
    for i = 1 : 6
        subplot(2, 3, i);
        plotSkel(pose3D{i}, 'r');
        title(sprintf('Pose 3D prediction - stage %d', i));
    end
    figure;
    set(gcf, 'Position', get(0,'Screensize'));
    for i = 1 : 6
        subplot(2, 3, i);
        imagesc(bodyPartLabeling{i}); 
        axis image; 
        axis off;
        title(sprintf('Body labeling prediction - stage %d', i));
    end

    figure;
    set(gcf, 'Position', get(0,'Screensize'));
    facealpha = 0.6; % for limb transparency
    truncate = zeros(1,model.np);
    for i = 1:6
        pred = pose2D{i};
        subplot(2,3,i);
        imshow(img);
        hold on;
        bodyHeight = max(pred(:,2)) - min(pred(:,2));
        plot_visible_limbs(model, facealpha, pred, truncate, bodyHeight/30);
        plot(pred(:,1), pred(:,2), 'k.', 'MarkerSize', bodyHeight/32);
        title(sprintf('Pose 2D prediction - stage %d', i));
    end;
end;
% keyboard;


% list = dir('/cluster/work/alin/DeepHumanReconstruction_Data/h36m_4000_data/Train/RGB/*.png');
% dimX = 0;
% dimY = 0;
% for i = 1 : numel(list);
%     img = imread(['/cluster/work/alin/DeepHumanReconstruction_Data/h36m_4000_data/Train/RGB/' list(i).name]);
%     dimX = dimX + size(img, 1);
%     dimY = dimY + size(img, 2);
% end
% dimX = dimX / numel(list);
% dimY = dimY / numel(list);

