clear, close all,clc
fd='verification_detect';
% fd = 'classification_detect';
datDir =['~/work/data/',fd,'/'];
savDir ='~/work/data/hog/';
load select_data2
%v_a_id
s_id=s_v_id;
len = length(s_id);
hogW = 8;
flen = 11*11*36;
dataset = zeros(10000,flen+1);
c=0;
im_n_c = zeros(len,5);
for i=1:len
    i
    for m=1:5
        imDir = [datDir,num2str(s_id(i)),'/',num2str(m),'/'];
        temp = dir([imDir,'*.jpg']);
%         length(temp)
%         im_n_c(i,m) = length(temp);
        for j=1:length(temp)
            im =imread([imDir,temp(j).name]);
            im_re = imresize(im,[100,100]);
            f = hogfeature36(double(im_re),hogW);
            c = c+1;
            dataset(c,:) = [f(:)',s_id(i)];
        end
    end
end

% save im_n_c im_n_c

dataset = dataset(1:c,:);
% %%
savefast([savDir,fd],'dataset');

