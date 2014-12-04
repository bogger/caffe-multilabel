clear,close all,clc
type ='surveillance2';
datDir=['~/work/data/',type];
savDir =['~/work/data/',type,'_resize'];




if ~exist(savDir,'dir')
    mkdir(savDir);
end
im_list = dir([datDir,'/*.jpg']);
im_n = length(im_list);
for k=1:im_n
    im = imread([datDir,'/',im_list(k).name]);
    im =imresize(im, [256,256]);
    imwrite(im,[savDir,'/',im_list(k).name]);
end

