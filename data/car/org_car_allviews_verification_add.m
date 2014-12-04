clear,close all,clc
datDir='~/work/data/verification_add_resize';

load select_data_add%s_id, s_v_id
rng(0);%rand seed
s_id = v_a_id;%for verification
n=length(s_id);

test_filename = 'verification_add_car';
full_l=256;
crop_l=227;

%256 -227


f_test = fopen([test_filename],'w');
% half train, half test
%train dup images [5/im_n] times

for i=1:n
   
        im_list = dir([datDir,'/',num2str(s_id(i)),'/*.jpg']);
        im_n = length(im_list);        


%%%%%%%%%%%%%%%%%%%%%%%%%%traditional testing%%%%%%%%%%%%%%%%%%%

        for k=1:im_n
            fprintf(f_test, [num2str(s_id(i)),'/',im_list(k).name]);
            fprintf(f_test,' %d\n',i-1);
        end

    
end

fclose(f_test);


%max(im_n,[],1)
