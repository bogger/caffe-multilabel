% clear, close all,clc
%load and test images from xcar
load car_final.mat
dtDir = '~/work/data/classification/';
saveDir ='~/work/data/classification_detect/';
load select_data2
%v_a_id
modules=5;
len = length(s_id);
for i=172:len
    i
    for m=1:modules
    imDir = [dtDir,num2str(s_id(i)),'/',num2str(m),'/'];
    temp = dir([imDir,'*.jpg']);

    for j=1:length(temp)
        im =imread([imDir,temp(j).name]);
        [bbox, bs] = process_one(im,model);
%         pause;
        outDir = [saveDir,num2str(s_id(i)),'/',num2str(m),'/'];
        if ~exist(outDir,'dir')
            mkdir(outDir);
        end
        outName = [outDir,temp(j).name];
        % save detected obj
        bbox = round(bbox(1:4));
        im_crop = im(bbox(2):bbox(4), bbox(1):bbox(3),:);
        imwrite(im_crop,outName);
        %pause;
    end
    end
end