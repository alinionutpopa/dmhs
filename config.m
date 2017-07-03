function param = config(use_gpu, device_number, modelID)

% CPU mode (0) or GPU mode (1)
% friendly warning: CPU mode may take a while
param.use_gpu = use_gpu;

% GPU device number (doesn't matter for CPU mode)
GPUdeviceNumber = device_number;

% Select model (default: 1)
% 1: 'Multitask Human3.6M - 6 Stages'

if(nargin < 3)
    param.modelID = 1;
else
    param.modelID = modelID;
end

% Scaling paramter: starting and ending ratio of person height to image
% height, and number of scales per octave
% warning: setting too small starting value on non-click mode will take
% large memory
param.octave = 6;


% WARNING! Adjust the path to your caffe accordingly!
caffepath = './code/caffe-pose-machines-cluster/matlab';

fprintf('You set your caffe in caffePath.cfg at: %s\n', caffepath);
addpath(caffepath);
caffe.reset_all();
if(param.use_gpu)
    fprintf('Setting to GPU mode, using device ID %d\n', GPUdeviceNumber);
    caffe.set_mode_gpu();
    caffe.set_device(GPUdeviceNumber);
else
    fprintf('Setting to CPU mode.\n');
    caffe.set_mode_cpu();
end

param.model(1).caffemodel = './model/net_iter_230000.caffemodel';
param.model(1).deployFile = './model/deploy.prototxt';

param.model(1).description = 'Pose 2D + Semantic Labeling + Pose3D on Human3.6M';
param.model(1).description_short = 'DMHS (H3.6M) - 6 Stages';
param.model(1).stage = 6;
param.model(1).boxsize = 368;
param.model(1).np = 14;
param.model(1).sigma = 21;
param.model(1).padValue = 128;
param.model(1).limbs = [13 14; 12 13; 11 12; 10 11; 9 13; 8 9; 7 8; 3 6; 5 6; 4 5; 3 2; 2 1];
param.model(1).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};

