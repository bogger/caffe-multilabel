clear,close all,clc
layers = {'fc8_door','fc8_seat','fc8_type'};
load im_n_v
load s_v_attr
load s_attr
sp_all = sum(im_n_v,1);


cls = size(im_n_v,1);
m_err_d = zeros(5,length(layers));
for i=1:5
    for l=1:length(layers)
        scores = load(['test_prob_car_',layers{l},'_',num2str(i)]);
    

        sp = sp_all(i);
        gt_label = zeros(sp,1);
        c=1;
        for j=1:cls
            gt_label(c:c+im_n_v(j,i)-1) = s_v_attr(j,l+2);
            c = c+im_n_v(j,i);
        end
        score = scores.feats(1:sp,:);
        [~,pred] = max(score,[],2);
        pred = pred - 1;
        %gt_var = var(gt_label)
        
        m_err_d(i,l) = mean(gt_label == pred);




    end

end

save m_err_d m_err_d