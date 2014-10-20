clear,close all,clc
datDir='~/work/data/classification_part_resize';
cropDir='~/work/data/classification_part_test_crop';
load select_data2%s_id, s_v_id
rng(0);%rand seed
%s_id = s_v_id;%for verification
n=length(s_id);
im_n = zeros(n,8);
train_filename='train_part';
test_filename = 'test_part_crop';
test_crop = 10;
full_l=256;
crop_l=227;
crop_s = full_l - crop_l + 1;
%256 -227
for i=1:8
    f_train(i) = fopen([train_filename,'_',num2str(i)],'w');
    f_test(i) = fopen([test_filename,'_',num2str(i)],'w');
end
% half train, half test
%train dup images [5/im_n] times
pos = randi(crop_s,n,2,test_crop);
rng(0);
for i=1:n
    for j=1:8
        im_list = dir([datDir,'/',num2str(s_id(i)),'/',num2str(j),'/*.jpg']);
        im_n(i,j) = length(im_list);
        train_n = round(im_n(i,j)/2);
        test_n = im_n(i,j) - train_n;
        p = randperm(im_n(i,j));%randomize
        %tp = randperm(test_n);
        dup_times = max(1,round(5/train_n));
        for k=1:train_n
           for dup=1:dup_times
            fprintf(f_train(j),[num2str(s_id(i)),'/',num2str(j),'/',im_list(p(k)).name]);
            fprintf(f_train(j),' %d\n',i-1);
           end
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%cropping testing%%%%%%%%%%%%%%%%%%%%%
        
%         
        for crop = 1:test_crop
        %for k=train_n+1:im_n(i,j)
            %%save cropped testing images
            k = train_n + 1 + mod(crop-1,test_n); 
            test_name = [num2str(s_id(i)),'/',num2str(j),'/',im_list(p(k)).name];
            im = imread([datDir,'/',test_name]);
            im_c = im(pos(i,1,crop):pos(i,1,crop) + crop_l - 1, ...
                pos(i,2,crop):pos(i,2,crop) + crop_l - 1,:);
            if ~exist([cropDir,'/',num2str(s_id(i)),'/',num2str(j)],'dir')
                mkdir([cropDir,'/',num2str(s_id(i)),'/',num2str(j)]);
            end
            crop_path = [num2str(s_id(i)),'/',num2str(j),'/', num2str(crop),'.jpg'];
            imwrite(im_c, [cropDir,'/',crop_path]);    
            fprintf(f_test(j), crop_path);
            fprintf(f_test(j),' %d\n',i-1);
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%traditional testing%%%%%%%%%%%%%%%%%%%
%
%         for k=train_n+1:im_n(i,j)
%             fprintf(f_test(j), [num2str(s_id(i)),'/',num2str(j),'/',im_list(p(k)).name]);
%             fprintf(f_test(j),' %d\n',i-1);
%         end

    end
end
for i=1:8
    fclose(f_train(i));
    fclose(f_test(i));
end
%max(im_n,[],1)
