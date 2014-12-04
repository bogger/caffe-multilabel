clear, close all,clc
%load and test images from xcar
load car_final.mat
dtDir = '~/work/data/verification_add/';
saveDir ='~/work/data/verfication_add_detect/';
load select_data_add
%v_a_id
len = length(v_a_id);
for i=1:len
    i
    imDir = [dtDir,num2str(v_a_id(i)),'/'];
    temp = dir([imDir,'*.jpg']);

    for j=1:length(temp)
        im =imread([imDir,temp(j).name]);
        [bbox, bs] = process_one(im,model);
%         pause;
        outDir = [saveDir,num2str(v_a_id(i)),'/'];
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