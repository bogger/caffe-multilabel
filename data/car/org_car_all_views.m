clear,close all,clc
datDir='~/work/data/classification_resize';
cropDir='~/work/data/classification_test_crop';
load select_data2%s_id, s_v_id
rng(0);%rand seed
%s_id = s_v_id;%for verification
n=length(s_id);
im_n = zeros(n,8);
train_filename='train_car';
test_filename = 'test_car_crop';
test_crop = 20;
full_l=256;
crop_l=227;
modules = 5;
crop_s = full_l - crop_l + 1;
%256 -227

    f_train = fopen([train_filename],'w');
    f_test = fopen([test_filename],'w');
% half train, half test
%train dup images [5/im_n] times
pos = randi(crop_s,n,2,test_crop);
rng(0);
for i=1:n
    for j=1:modules % 5 views
        im_list = dir([datDir,'/',num2str(s_id(i)),'/',num2str(j),'/*.jpg']);
        im_n(i,j) = length(im_list);
        train_n = round(im_n(i,j)/2);
        test_n = im_n(i,j) - train_n;
        p = randperm(im_n(i,j));%randomize
        %tp = randperm(test_n);
        dup_times = max(1,round(10/train_n));
        for k=1:train_n
           for dup=1:dup_times
            fprintf(f_train,[num2str(s_id(i)),'/',num2str(j),'/',im_list(p(k)).name]);
            fprintf(f_train,' %d\n',i-1);
           end
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%cropping testing%%%%%%%%%%%%%%%%%%%%%
        
%         
        for crop = 1:test_crop
        %for k=train_n+1:im_n(i,j)
            %%save cropped testing images
 

            crop_path = [num2str(s_id(i)),'/',num2str(j),'/', num2str(crop),'.jpg'];
            fprintf(f_test, crop_path);
            fprintf(f_test,' %d\n',i-1);
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%traditional testing%%%%%%%%%%%%%%%%%%%
%
%         for k=train_n+1:im_n(i,j)
%             fprintf(f_test(j), [num2str(s_id(i)),'/',num2str(j),'/',im_list(p(k)).name]);
%             fprintf(f_test(j),' %d\n',i-1);
%         end

    end
end

fclose(f_train);
fclose(f_test);

%max(im_n,[],1)
