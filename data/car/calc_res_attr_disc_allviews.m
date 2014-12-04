clear,close all,clc
layers = {'fc8_door','fc8_seat','fc8_type'};
load im_n_v
load s_v_attr
load s_attr
sp_all = sum(im_n_v(:));%sum(im_n_v,1);
im_n_t = sum(im_n_v,2);

mean_tr(1) = mean(s_attr(:,1));
std_tr(1) = std(s_attr(:,1));
mean_tr(2) = mean(s_attr(:,2));
std_tr(2) = std(s_attr(:,2));

s_v_attr(:,1) = (s_v_attr(:,1) - mean_tr(1))/std_tr(1);
s_v_attr(:,2) = (s_v_attr(:,2) - mean_tr(2))/std_tr(2);
cls = size(im_n_t,1);
m_err_d = zeros(3,1);

    for l=1:length(layers)
        scores = load(['test_prob_car_',layers{l}]);
    

        sp = sp_all;
        gt_label = zeros(sp,1);

        c=1;
        for j=1:cls
            gt_label(c:c+im_n_t(j)-1) = s_v_attr(j,l+2);
            c = c+im_n_t(j);
        end
        score = scores.feats(1:sp,:);
        [~,pred] = max(score,[],2);
        pred = pred - 1;
        %gt_var = var(gt_label)
        
        m_err_d(l) = mean(gt_label == pred);



    end



save m_err_disc_allviews m_err_d