% clear,close all,clc
% scores = cell(8,1);
% for i=1:8
%     scores{i} = load(['test_prob_part_',num2str(i)]);
% end
sp = 4420;
cls = 442;
sp_per_cls = 10;
gt_label = zeros(sp,1);
for i=1:cls
    gt_label((i-1)*sp_per_cls+1:i*sp_per_cls) = i;
end
ac=zeros(8,2);

votes = zeros(sp,cls);
ord = [1:sp];
for i=1:8
    [~,sc_sort] = sort(scores{i}.feats,2,'descend');
    vote = sc_sort(:,1);
    ac(i,1) = sum(vote==gt_label)/sp;
    ac(i,2) = sum(any(sc_sort(:,1:5)==repmat(gt_label,1,5),2))/sp;
    for j=1:sp
        votes(j,vote(j)) = votes(j,vote(j))+1;
    end
end

[~,pred] = sort(votes,2,'descend');
ac_vote = sum(pred(:,1)==gt_label)/sp;
ac_vote_t5 = sum(any(pred(:,1:5)==repmat(gt_label,1,5),2))/sp;

