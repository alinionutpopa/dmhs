function [predictions, network_output] = applyModelPreloadedNet_DMHS(net, oriImg, param, rectangle, multiplier_3D, multiplier_sem)
%% Select model and other parameters from param
model = param.model(param.modelID);
boxsize = model.boxsize;
np = model.np;
nstage = model.stage;

network_output = cell(3, 1);
predictions = cell(3, 1);

%% Apply multitask network model
% set the center and roughly scale range (overwrite the config!) according to the rectangle
x_start = max(rectangle(1), 1);
x_end = min(rectangle(1)+rectangle(3), size(oriImg,2));
y_start = max(rectangle(2), 1);
y_end = min(rectangle(2)+rectangle(4), size(oriImg,1));
center = [(x_start + x_end)/2, (y_start + y_end)/2];

octave = param.octave;
middle_range = (y_end - y_start) / size(oriImg,1) * 1.2;
starting_range = middle_range * 0.8;
ending_range = middle_range * 3.0;

starting_scale = boxsize/(size(oriImg,1)*ending_range);
ending_scale = boxsize/(size(oriImg,1)*starting_range);
multiplier_2D = 2.^(log2(starting_scale):(1/octave):log2(ending_scale));

multiplier = union(multiplier_2D, union(multiplier_3D, multiplier_sem));

buffer_score = cell(nstage, 3, length(multiplier));
score_2D = cell(nstage, length(multiplier_2D));
score_3D = cell(nstage, length(multiplier_3D));
score_sem = cell(nstage, length(multiplier_sem));
pad = cell(1, length(multiplier));
ori_size = cell(1, length(multiplier));

% net = caffe.Net(model.deployFile, model.caffemodel, 'test');
% change outputs to enable visualizing stagewise results
% note this is why we keep out own copy of m-files of caffe wrapper

for m = 1:length(multiplier)
    scale = multiplier(m);
    imageToTest = imresize(oriImg, scale);
    ori_size{m} = size(imageToTest);
    center_s = center * scale;
    [imageToTest, pad{m}] = padAround(imageToTest, boxsize, center_s, model.padValue);
    
    imageToTest = preprocess(imageToTest, 0.5, param);
    buffer_score(:,:,m) = applyDNN(imageToTest, net);
    pool_time = size(imageToTest,1) / size(buffer_score{1,2,m},1); % stride-8
    % make heatmaps into the size of original image according to pad and scale
    % this part can be optimizied if needed
    
    buffer_score(:,2,m) = cellfun(@(x) imresize(x, pool_time), buffer_score(:,2,m), 'UniformOutput', false);
    buffer_score(:,2,m) = cellfun(@(x) resizeIntoScaledImg(x, pad{m}), buffer_score(:,2,m), 'UniformOutput', false);
    buffer_score(:,2,m) = cellfun(@(x) imresize(x, [size(oriImg,2) size(oriImg,1)]), buffer_score(:,2,m), 'UniformOutput', false);
    buffer_score(:,3,m) = cellfun(@(x) resizeIntoScaledImg(x, pad{m}), buffer_score(:,3,m), 'UniformOutput', false);
    buffer_score(:,3,m) = cellfun(@(x) imresize(x, [size(oriImg,2) size(oriImg,1)]), buffer_score(:,3,m), 'UniformOutput', false);
    if (~sum(find(multiplier_3D == scale))==0)
        idx = find(multiplier_3D == scale);
        score_3D(:,idx) = buffer_score(:,1,m); 
    end;
    if (~sum(find(multiplier_sem == scale))==0)
        idx = find(multiplier_sem == scale);
        score_sem(:,idx) = buffer_score(:,3,m); 
    end;
    if (~sum(find(multiplier_2D == scale))==0)
        idx = find(multiplier_2D == scale);
        score_2D(:,idx) = buffer_score(:,2,m); 
    end;
    
end

network_output{1} = score_2D;
network_output{2} = score_sem;
network_output{3} = score_3D;

%% Process 2D POSE
final_score = cell(nstage, 1);
for s = 1:nstage
    final_score{s} = zeros(size(score_2D{s,1}));
    for m = 1:size(score_2D,2)
        final_score{s} = final_score{s} + score_2D{s,m};
    end
    
    prediction = zeros(np,2);
    for j = 1:np
        [prediction(np-j+1,1), prediction(np-j+1,2)] = findMaximum(final_score{s}(:,:,j));
    end
    predictions{1}{s} = prediction;
end

%% Process BODY PART LABELING
final_score_sem = cell(nstage, 1);
for s = 1:nstage
    final_score_sem{s} = zeros(size(score_sem{s,1}));
    for m = 1:size(score_sem,2)
         score_sem{s,m} = softmax_caffe(score_sem{s,m});
         final_score_sem{s} = final_score_sem{s} + score_sem{s,m};
    end
    heatMaps = permute(final_score_sem{s}, [2 1 3]);
    heatMaps = heatMaps / size(score_sem,2);
    [~, predictions{2}{s}] = max(heatMaps(:, :, 1:end-1), [], 3);
    predictions{2}{s} = uint8(predictions{2}{s} - 1);
end

%% Process 3D POSE
for s = 1:nstage
    est = zeros(17, 3);
    for m = 1:length(multiplier_3D)
        temp = score_3D{s, m};
        est = est + reshape(temp, [17 3]);
    end
    predictions{3}{s} = est / length(multiplier_3D);
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
    scores = cell(6, 3);
    for s = 1:6
        string_to_search_v1 = sprintf('pose_3D_pose_regress_%d', s);


        blob_id = net.name2blob_index(string_to_search_v1);
        scores{s, 1} = net.blob_vec(blob_id).get_data();

        if(s >= 2)
            string_to_search_pose2D = sprintf('Mconv5_stage%d', s);
        else
            string_to_search_pose2D = sprintf('conv7_stage1');
        end
        blob_id = net.name2blob_index(string_to_search_pose2D);
        scores{s,2} = net.blob_vec(blob_id).get_data();
        
        if(s >= 2)
            string_to_search_sem = sprintf('Mconv5_stage%d_sem', s);
        else
            string_to_search_sem = sprintf('conv7_stage1_sem');
        end
        blob_id = net.name2blob_index(string_to_search_sem);
        scores2 = net.blob_vec(blob_id).get_data();
        scores{s,3} = Deconvolution(scores2);
    end
    
function out = Deconvolution(scores)
K = [0.0039    0.0117    0.0195    0.0273    0.0352    0.0430    0.0508    0.0586    0.0586    0.0508    0.0430    0.0352    0.0273    0.0195    0.0117    0.0039
    0.0117    0.0352    0.0586    0.0820    0.1055    0.1289    0.1523    0.1758    0.1758    0.1523    0.1289    0.1055    0.0820    0.0586    0.0352    0.0117
    0.0195    0.0586    0.0977    0.1367    0.1758    0.2148    0.2539    0.2930    0.2930    0.2539    0.2148    0.1758    0.1367    0.0977    0.0586    0.0195
    0.0273    0.0820    0.1367    0.1914    0.2461    0.3008    0.3555    0.4102    0.4102    0.3555    0.3008    0.2461    0.1914    0.1367    0.0820    0.0273
    0.0352    0.1055    0.1758    0.2461    0.3164    0.3867    0.4570    0.5273    0.5273    0.4570    0.3867    0.3164    0.2461    0.1758    0.1055    0.0352
    0.0430    0.1289    0.2148    0.3008    0.3867    0.4727    0.5586    0.6445    0.6445    0.5586    0.4727    0.3867    0.3008    0.2148    0.1289    0.0430
    0.0508    0.1523    0.2539    0.3555    0.4570    0.5586    0.6602    0.7617    0.7617    0.6602    0.5586    0.4570    0.3555    0.2539    0.1523    0.0508
    0.0586    0.1758    0.2930    0.4102    0.5273    0.6445    0.7617    0.8789    0.8789    0.7617    0.6445    0.5273    0.4102    0.2930    0.1758    0.0586
    0.0586    0.1758    0.2930    0.4102    0.5273    0.6445    0.7617    0.8789    0.8789    0.7617    0.6445    0.5273    0.4102    0.2930    0.1758    0.0586
    0.0508    0.1523    0.2539    0.3555    0.4570    0.5586    0.6602    0.7617    0.7617    0.6602    0.5586    0.4570    0.3555    0.2539    0.1523    0.0508
    0.0430    0.1289    0.2148    0.3008    0.3867    0.4727    0.5586    0.6445    0.6445    0.5586    0.4727    0.3867    0.3008    0.2148    0.1289    0.0430
    0.0352    0.1055    0.1758    0.2461    0.3164    0.3867    0.4570    0.5273    0.5273    0.4570    0.3867    0.3164    0.2461    0.1758    0.1055    0.0352
    0.0273    0.0820    0.1367    0.1914    0.2461    0.3008    0.3555    0.4102    0.4102    0.3555    0.3008    0.2461    0.1914    0.1367    0.0820    0.0273
    0.0195    0.0586    0.0977    0.1367    0.1758    0.2148    0.2539    0.2930    0.2930    0.2539    0.2148    0.1758    0.1367    0.0977    0.0586    0.0195
    0.0117    0.0352    0.0586    0.0820    0.1055    0.1289    0.1523    0.1758    0.1758    0.1523    0.1289    0.1055    0.0820    0.0586    0.0352    0.0117
    0.0039    0.0117    0.0195    0.0273    0.0352    0.0430    0.0508    0.0586    0.0586    0.0508    0.0430    0.0352    0.0273    0.0195    0.0117    0.0039];
    pad = 4;
    stride = 8;
    no_classes = 26;
    k = size(K, 1);
    n = size(scores, 1);
    img_size = 2*pad + 1 + (n-1)*stride;
    out_size = k + (n-1)*stride - 2*pad;
    out = zeros(out_size, out_size, no_classes);
    img = zeros(img_size, img_size, no_classes);
    img(pad+1:stride:end-pad, pad+1:stride:end-pad, :) = scores;
%     pad = (size(temp, 1) - out_size) / 2;
    pad = 8;
    for c = 1:no_classes
        temp = conv2(squeeze(img(:,:,c)), K);
        out(:,:,c) = temp(pad+1:end-pad, pad+1:end-pad);
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

function [x,y] = findMaximum(map)
    [~,i] = max(map(:));
    [x,y] = ind2sub(size(map), i);
    
function score = resizeIntoScaledImg(score, pad)
    np = size(score,3)-1;
    score = permute(score, [2 1 3]);
    if(pad(1) < 0)
        padup = cat(3, zeros(-pad(1), size(score,2), np), ones(-pad(1), size(score,2), 1));
        score = [padup; score]; % pad up
    else
        score(1:pad(1),:,:) = []; % crop up
    end
    
    if(pad(2) < 0)
        padleft = cat(3, zeros(size(score,1), -pad(2), np), ones(size(score,1), -pad(2), 1));
        score = [padleft score]; % pad left
    else
        score(:,1:pad(2),:) = []; % crop left
    end
    
    if(pad(3) < 0)
        paddown = cat(3, zeros(-pad(3), size(score,2), np), ones(-pad(3), size(score,2), 1));
        score = [score; paddown]; % pad down
    else
        score(end-pad(3)+1:end, :, :) = []; % crop down
    end
    
    if(pad(4) < 0)
        padright = cat(3, zeros(size(score,1), -pad(4), np), ones(size(score,1), -pad(4), 1));
        score = [score padright]; % pad right
    else
        score(:,end-pad(4)+1:end, :) = []; % crop right
    end
    score = permute(score, [2 1 3]);
    
function label = produceCenterLabelMap(im_size, x, y, sigma)
    % this function generates a gaussian peak centered at position (x,y)
    % it is only for center map in testing
    [X,Y] = meshgrid(1:im_size(1), 1:im_size(2));
    X = X - x;
    Y = Y - y;
    D2 = X.^2 + Y.^2;
    Exponent = D2 ./ 2.0 ./ sigma ./ sigma;
    label{1} = exp(-Exponent);