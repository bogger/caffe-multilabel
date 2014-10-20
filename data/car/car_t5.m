clear,close all,clc
scores = cell(5,1);
for i=1:5
    scores{i} = load(['test_prob_car_',num2str(i)]);
end
sp = 8840;
cls = 442;
sp_per_cls = 20;
gt_label = zeros(sp,1);
for i=1:cls
    gt_label((i-1)*sp_per_cls+1:i*sp_per_cls) = i;
end
ac=zeros(5,2);


for i=1:5
    [~,sc_sort] = sort(scores{i}.feats,2,'descend');
    ac(i,1) = sum(sc_sort(:,1)==gt_label)/sp;
    ac(i,2) = sum(any(sc_sort(:,1:5)==repmat(gt_label,1,5),2))/sp;

end


