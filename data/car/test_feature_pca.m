clear,close all,clc
%pca->joint bayesian->test

f =load('test_feature');
%load verification_add_feature
load im_n_c
sp=44200;

feature = f.feats(1:sp,:);

dim = 40;
[feature_d, Vp, f_m] = my_pca(feature,dim);

save('test_feature_d','feature_d');