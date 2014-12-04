clear,close all,clc
scores = cell(5,1);

scores = load(['test_prob_car']);

sp = 44200;
sp_per_m = 8840;
cls = 442;
sp_per_cls = 100;
gt_label = zeros(sp,1);

for i=1:cls
    gt_label((i-1)*sp_per_cls+1 :i*sp_per_cls ) = i;
end

ac=zeros(1,2);



[~,sc_sort] = sort(scores.feats,2,'descend');
ac(1) = sum(sc_sort(:,1)==gt_label)/sp;
ac(2) = sum(any(sc_sort(:,1:5)==repmat(gt_label,1,5),2))/sp;


