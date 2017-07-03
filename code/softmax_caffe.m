function [ B ] = softmax_caffe(A)

B = exp(A);
B = B ./ repmat(sum(B, 3) + eps, [1 1 size(A, 3)]);



end

