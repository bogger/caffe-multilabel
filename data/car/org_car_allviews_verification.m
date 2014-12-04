clear,close all,clc
datDir='~/work/data/verification_resize';

load select_data2%s_id, s_v_id
rng(0);%rand seed
s_id = s_v_id;%for verification
n=length(s_id);

test_filename = 'verification_car';
test_crop = 20;
full_l=256;
crop_l=227;
modules = 5;
crop_s = full_l - crop_l + 1;
%256 -227


f_test = fopen([test_filename],'w');
% half train, half test
%train dup images [5/im_n] times

for i=1:n
    for j=1:modules
        im_list = dir([datDir,'/',num2str(s_id(i)),'/',num2str(j),'/*.jpg']);
        im_n = length(im_list);        


%%%%%%%%%%%%%%%%%%%%%%%%%%traditional testing%%%%%%%%%%%%%%%%%%%

        for k=1:im_n
            fprintf(f_test, [num2str(s_id(i)),'/',num2str(j),'/',im_list(k).name]);
            fprintf(f_test,' %d\n',i-1);
        end

    end
end

fclose(f_test);


%max(im_n,[],1)
