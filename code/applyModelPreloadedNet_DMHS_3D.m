function predictions = applyModelPreloadedNet_DMHS_3D(net, oriImg, param, rectangle, multiplier_3D)
%% Select model and other parameters from param
model = param.model(param.modelID);
boxsize = model.boxsize;
np = model.np;
nstage = model.stage;

predictions = cell(1, 1);

%% Apply multitask network model
% set the center and roughly scale range (overwrite the config!) according to the rectangle
x_start = max(rectangle(1), 1);
x_end = min(rectangle(1)+rectangle(3), size(oriImg,2));
y_start = max(rectangle(2), 1);
y_end = min(rectangle(2)+rectangle(4), size(oriImg,1));
center = [(x_start + x_end)/2, (y_start + y_end)/2];

buffer_score = cell(nstage, 3, length(multiplier_3D));
score_3D = cell(nstage, length(multiplier_3D));
pad = cell(1, length(multiplier_3D));
ori_size = cell(1, length(multiplier_3D));

% net = caffe.Net(model.deployFile, model.caffemodel, 'test');
% change outputs to enable visualizing stagewise results
% note this is why we keep out own copy of m-files of caffe wrapper

for m = 1:length(multiplier_3D)
    scale = multiplier_3D(m);
    imageToTest = imresize(oriImg, scale);
    ori_size{m} = size(imageToTest);
    center_s = center * scale;
    [imageToTest, pad{m}] = padAround(imageToTest, boxsize, center_s, model.padValue);
    
    imageToTest = preprocess(imageToTest, 0.5, param);
    buffer_score(:,m) = applyDNN(imageToTest, net);
    
    if (~sum(find(multiplier_3D == scale))==0)
        idx = find(multiplier_3D == scale);
        score_3D(:,idx) = buffer_score(:,m); 
    end;
    
end

%% Process 3D POSE
for s = 1:nstage
    est = zeros(17, 3);
    for m = 1:length(multiplier_3D)
        temp = score_3D{s, m};
        est = est + reshape(temp, [17 3]);
    end
    predictions{s} = est / length(multiplier_3D);
end

function img_out = preprocess(img, mean, param)
    img_out = double(img)/256;  
    img_out = double(img_out) - mean;
    img_out = permute(img_out, [2 1 3]);
    
    img_out = img_out(:,:,[3 2 1]); % BGR for opencv training in caffe !!!!!
    boxsize = param.model(param.modelID).boxsize;
    centerMapCell = produceCenterLabelMap([boxsize boxsize], boxsize/2, boxsize/2, param.model(param.modelID).sigma);
    img_out(:,:,4) = centerMapCell{1};
    
function scores = applyDNN(images, net)
    input_data = {single(images)};
    % do forward pass to get scores
    % scores are now Width x Height x Channels x Num
    net.forward(input_data);
    scores = cell(6, 1);
    for s = 1:6
        string_to_search_v1 = sprintf('pose_3D_pose_regress_%d', s);


        blob_id = net.name2blob_index(string_to_search_v1);
        scores{s} = net.blob_vec(blob_id).get_data();
    end

function [img_padded, pad] = padAround(img, boxsize, center, padValue)
    center = round(center);
    h = size(img, 1);
    w = size(img, 2);
    pad(1) = boxsize/2 - center(2); % up
    pad(3) = boxsize/2 - (h-center(2)); % down
    pad(2) = boxsize/2 - center(1); % left
    pad(4) = boxsize/2 - (w-center(1)); % right
    
    pad_up = repmat(img(1,:,:), [pad(1) 1 1])*0 + padValue;
    img_padded = [pad_up; img];
    pad_left = repmat(img_padded(:,1,:), [1 pad(2) 1])*0 + padValue;
    img_padded = [pad_left img_padded];
    pad_down = repmat(img_padded(end,:,:), [pad(3) 1 1])*0 + padValue;
    img_padded = [img_padded; pad_down];
    pad_right = repmat(img_padded(:,end,:), [1 pad(4) 1])*0 + padValue;
    img_padded = [img_padded pad_right];
    
    center = center + [max(0,pad(2)) max(0,pad(1))];

    img_padded = img_padded(center(2)-(boxsize/2-1):center(2)+boxsize/2, center(1)-(boxsize/2-1):center(1)+boxsize/2, :); %cropping if needed

function label = produceCenterLabelMap(im_size, x, y, sigma)
    % this function generates a gaussian peak centered at position (x,y)
    % it is only for center map in testing
    [X,Y] = meshgrid(1:im_size(1), 1:im_size(2));
    X = X - x;
    Y = Y - y;
    D2 = X.^2 + Y.^2;
    Exponent = D2 ./ 2.0 ./ sigma ./ sigma;
    label{1} = exp(-Exponent);