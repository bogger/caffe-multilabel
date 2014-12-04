clear, close all,clc
% fd='verification_detect';
fd = 'verification_add_detect';
datDir =['~/work/data/',fd,'/'];
savDir ='~/work/data/hog/';
load select_data_add
%v_a_id
s_id = double(v_a_id); %int32 format of v_a_id cause a bug in "dataset"
len = length(s_id);
hogW = 8;
flen = 11*11*36;
dataset = zeros(30000,flen+1);
c=0;
for i=1:len
    i
%     for m=1:5
        imDir = [datDir,num2str(s_id(i)),'/'];
        temp = dir([imDir,'*.jpg']);
%         length(temp)
        for j=1:length(temp)
            im =imread([imDir,temp(j).name]);
            im_re = imresize(im,[100,100]);
            f = hogfeature36(double(im_re),hogW);
%             pause;
            c = c+1;
            dataset(c,:) = [f(:)',s_id(i)];
%             pause;
        end
%     end
end
dataset = dataset(1:c,:);
%%
savefast([savDir,fd],'dataset');