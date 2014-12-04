clear,close all,clc
datDir='~/work/data/surveillance2_resize';

rng(0);%rand seed

test_filename = 'surveillance_car';

full_l=256;
crop_l=227;
crop_s = full_l - crop_l + 1;
%256 -227


f_test = fopen([test_filename],'w');
% half train, half test
%train dup images [5/im_n] times



im_list = dir([datDir,'/*.jpg']);
im_n = length(im_list);        


%%%%%%%%%%%%%%%%%%%%%%%%%%traditional testing%%%%%%%%%%%%%%%%%%%

for k=1:im_n
    fprintf(f_test, im_list(k).name);
    fprintf(f_test,' %d\n',0);
end

    


fclose(f_test);
