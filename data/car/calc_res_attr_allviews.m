clear,close all,clc
layers = {'fc8_speed','fc8_disl'};
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
m_err = zeros(2,2);

    for l=1:length(layers)
        scores = load(['test_prob_car_',layers{l}]);
    

        sp = sp_all;
        gt_label = zeros(sp,1);
        c=1;
        for j=1:cls
            gt_label(c:c+im_n_t(j)-1) = s_v_attr(j,l);
            c = c+im_n_t(j);
        end
        score = scores.feats(1:sp);
        %gt_var = var(gt_label)
        
        m_err(l,1) = mean(abs(gt_label))*std_tr(l);
        m_err(l,2) = mean(abs(score - gt_label))*std_tr(l);



    end



save m_err_allviews m_err