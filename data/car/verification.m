clear,close all,clc
%pca->joint bayesian->test

f =load('verification_feature');
%load verification_add_feature
load im_n_v
n = sum(im_n_v(:));
model_n = size(im_n_v,1);
modules = 5;
feature = f.feats(1:n,:);

dim = 20;
[feature_d, Vp, f_m] = my_pca(feature,dim);
% C2=feature_d'*feature_d;
% eig_diff = diag(C2) - d(idx(1:dim));
gt_label = zeros(n,1);
c=0;
for i=1:model_n
    for j=1:modules
        gt_label(c+1:c+im_n_v(i,j)) = i;
        c = c+im_n_v(i,j);
    end
end
%%
% model = trainBayesian2(feature_d,gt_label);
% save jb_model model
%%
load jb_model
ft = load('verification_add_feature');
load im_n_a
n_a = sum(im_n_a(:));
feature_t = ft.feats(1:n_a,:);
feature_t = bsxfun(@minus, feature_t, f_m);
feature_td = feature_t * Vp;

n_match = 10000;
n_unmatch=10000;
[pairs_match,pairs_unmatch] = gen_verif_pairs(im_n_a,n_match,n_unmatch);
% load verification_pair

score_match = verifyBayesian(model,feature_td(pairs_match(:,1),:),feature_td(pairs_match(:,2),:));
score_unmatch = verifyBayesian(model,feature_td(pairs_unmatch(:,1),:),feature_td(pairs_unmatch(:,2),:));


%% draw curve
thresh=-50:0.1:50;
len = length(thresh);
fp=zeros(len,1);
recall=zeros(len,1);
for i=1:length(thresh)
    recall(i) = sum(score_match > thresh(i))/n_match;
    fp(i) = sum(score_unmatch > thresh(i))/n_unmatch;
end
figure;
plot(fp,recall);
% semilogx(fp,recall);
save cnn_jb_roc fp recall
% set thresh to 0
th=-4;
ac = (sum(score_match > th) + sum(score_unmatch <=th))/(n_match +n_unmatch)
save verif_info pairs_match pairs_unmatch score_match score_unmatch