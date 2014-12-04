clear,close all,clc
load select_data2
scores = load(['sv_car2']);

sp = 20;

cls = 442;



ac=zeros(1,2);
datDir='~/work/data/surveillance2_resize';
im_list = dir([datDir,'/*.jpg']);


[~,sc_sort] = sort(scores.feats(1:sp,:),2,'descend');
% ac(1) = sum(sc_sort(:,1)==gt_label)/sp;
% ac(2) = sum(any(sc_sort(:,1:5)==repmat(gt_label,1,5),2))/sp;
sv_pred_id = s_id(sc_sort(:,1));
save sv_pred2 sv_pred_id im_list