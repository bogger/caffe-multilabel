clear,close all,clc
layers = {'fc8_speed','fc8_disl'};
load im_n_v
load s_v_attr
load s_attr
sp_all = sum(im_n_v,1);

mean_tr(1) = mean(s_attr(:,1));
std_tr(1) = std(s_attr(:,1));
mean_tr(2) = mean(s_attr(:,2));
std_tr(2) = std(s_attr(:,2));

s_v_attr(:,1) = (s_v_attr(:,1) - mean_tr(1))/std_tr(1);
s_v_attr(:,2) = (s_v_attr(:,2) - mean_tr(2))/std_tr(2);
cls = size(im_n_v,1);
m_err = zeros(5,2,2);
for i=1:5
    for l=1:length(layers)
        scores = load(['test_prob_car_',layers{l},'_',num2str(i)]);
    

        sp = sp_all(i);
        gt_label = zeros(sp,1);
        c=1;
        for j=1:cls
            gt_label(c:c+im_n_v(j,i)-1) = s_v_attr(j,l);
            c = c+im_n_v(j,i);
        end
        score = scores.feats(1:sp);
        %gt_var = var(gt_label)
        
        m_err(i,l,1) = mean(abs(gt_label))*std_tr(l);
        m_err(i,l,2) = mean(abs(score - gt_label))*std_tr(l);



    end

end

save m_err m_err