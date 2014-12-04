clear, close all,clc
load im_n_v
load im_n_c
im_n = [im_n_c;im_n_v];
model_n = size(im_n,1);



n_match = 100000;
n_unmatch= 100000;
[pairs_match, pairs_unmatch] = gen_verif_pairs(im_n,n_match,n_unmatch);



%%
datDir ='~/work/data/hog/classification_detect';
datDir2 = '~/work/data/hog/verification_detect';
c_data = load(datDir);
v_data = load(datDir2);

combine_data = [c_data.dataset(:,1:end-1);v_data.dataset(:,1:end-1)];
% combine_data = single(combine_data);
clear c_data v_data
%pca
% dim=200;
% [feature, Vp, f_m] = my_pca(combine_data,dim);
% f_max = max(feature(:));
% f_min = min(feature(:));
% feature = (feature - f_min)/(f_max-f_min);
%no pca
feature = combine_data*5;
clear combine_data
% svm_data = zeros(f_len +1 ,n_match + n_unmatch);
%%

svm_data = [feature(pairs_match(:,1),:), feature(pairs_match(:,2),:);...     
            feature(pairs_unmatch(:,1),:),feature(pairs_unmatch(:,2),:)];
% clear combine_data
label = [ones(n_match,1);zeros(n_unmatch,1)];
%train a linear svm
model = train(label,sparse(svm_data),'-s 2 -c 1');
% model = svmtrain(label, svm_data,'-t 2');
%save model
save svm_model model
%%
%parse testing matrix
% load svm_model
testDir = '~/work/data/hog/verification_add_detect';
a_data = load(testDir);
a_data = a_data.dataset(:,1:end-1);%normalize to 0~1
%pca
feature_t = a_data * Vp;
feature_t = (feature_t - f_min)/(f_max - f_min);
%no pca
% feature_t = a_data*5;
% load verification_pair
load im_n_a
n_match=10000;
n_unmatch=10000;
[pairs_match,pairs_unmatch] = gen_verif_pairs(im_n_a,n_match,n_unmatch);


test_data = [feature_t(pairs_match(:,1),:),feature_t(pairs_match(:,2),:);...
           feature_t(pairs_unmatch(:,1),:), feature_t(pairs_unmatch(:,2),:)];
test_label = [ones(n_match,1);zeros(n_unmatch,1)];

[preds, ac, dec_values] = predict(test_label, sparse(test_data),model);

%%
% [preds, ac, dec_values] = svmpredict(test_label, test_data, model);
thresh=-7:0.01:7;
len = length(thresh);
fp=zeros(len,1); 
recall=zeros(len,1);
for i=1:length(thresh)
    recall(i) = sum(dec_values(1:n_match) > thresh(i))/n_match;
    fp(i) = sum(dec_values(n_match+1:n_match+n_unmatch) > thresh(i))/n_unmatch;
end
close all;
plot(fp,recall);
save  hog_svm_roc fp recall
