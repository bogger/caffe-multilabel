% clear,close all,clc
% %svm->test
% 
f =load('verification_feature');
%load verification_add_feature
load im_n_v
n = sum(im_n_v(:));
model_n = size(im_n_v,1);
modules = 5;
feature = f.feats(1:n,:);
% f_m = mean(feature,1);
% feature = bsxfun(@minus,feature,f_m);
% C = feature' * feature;
% normalize feature to 0~1
dim = 20;
% [feature, Vp, f_m] = my_pca(feature,dim);

%no pca
f_m = max(feature(:));
feature = feature./f_m;

% [V,D]= eig(C);
% [~,idx] = sort(diag(D),'descend');
% d = diag(D);
% D = diag(d(idx));
% V = V(:,idx);

n_match = 50000;
n_unmatch= 50000;
[pairs_match, pairs_unmatch] = gen_verif_pairs(im_n_v, n_match, n_unmatch);



%%
train_data =[feature(pairs_match(:,1),:),feature(pairs_match(:,2),:);...
    feature(pairs_unmatch(:,1),:),feature(pairs_unmatch(:,2),:)];
train_label=[ones(n_match,1);zeros(n_unmatch,1)];
model = train(train_label,sparse(train_data),'-s 2 -c 1');

%%

ft = load('verification_add_feature');
load im_n_a
n_a = sum(im_n_a(:));
model_n_a = size(im_n_a,1);
feature_t = ft.feats(1:n_a,:);
% no pca
feature_t = feature_t./f_m;
% feature_t = bsxfun(@minus, feature_t, f_m);
% feature_t = feature_t*Vp;
% gt_label_t = zeros(n_a,1);
% m_offset = zeros(model_n_a,1);
% im_t = sum(im_n_a,2);
% c=0;
% for i=1:model_n_a
%     m_offset(i) = c;
%     gt_label_t(c+1:c+im_t(i)) = i;
%     c = c+im_t(i);
%     
% 
% end


%%
   
% load verification_pair
load cnn_svm_model
n_match = 10000;
n_unmatch=10000;
[pairs_match,pairs_unmatch] = gen_verif_pairs(im_n_a,n_match,n_unmatch);

test_data = [feature_t(pairs_match(:,1),:),feature_t(pairs_match(:,2),:);...
             feature_t(pairs_unmatch(:,1),:),feature_t(pairs_unmatch(:,2),:)];
test_label = [ones(n_match,1);zeros(n_unmatch,1)];
[preds, ac, dec_values] = predict(test_label,sparse(test_data),model);


%% draw curve
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
save  cnn_svm_roc fp recall
% figure;
% semilogx(fp,recall);
% % set thresh to 0
% ac = (sum(score_match > 0) + sum(score_unmatch <=0))/(n_match +n_unmatch);

