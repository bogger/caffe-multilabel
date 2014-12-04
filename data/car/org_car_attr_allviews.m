clear,close all,clc

datDir='~/work/data/classification_resize';
testDir='~/work/data/verification_resize';
load select_data2%s_id, s_v_id
load s_attr
load s_v_attr
rng(0);%rand seed
%s_id = s_v_id;%for verification
n=length(s_id);
n_v = length(s_v_id);
im_n = zeros(n,5);
im_n_v = zeros(n_v,5);
train_filename='train_car_attr_cont';
test_filename = 'test_car_attr_cont';
%regulize attr
mean1 = mean(s_attr(:,1));
std1 = std(s_attr(:,1));
mean2 = mean(s_attr(:,2));
std2 = std(s_attr(:,2));

s_attr(:,1) = (s_attr(:,1)-mean1)/std1;
s_attr(:,2) = (s_attr(:,2)-mean2)/std2;

s_v_attr(:,1) =(s_v_attr(:,1)-mean1)/std1;
s_v_attr(:,2) = (s_v_attr(:,2)-mean2)/std2;
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
% pos = randi(crop_s,n,2,test_crop);
rng(0);

%%%%%%% organize training data %%%%%%%%%%%%%%
for i=1:n
    for j=1:modules % 5 views
        im_list = dir([datDir,'/',num2str(s_id(i)),'/',num2str(j),'/*.jpg']);
        im_n(i,j) = length(im_list);      
                
        for k=1:im_n(i,j)
          
            fprintf(f_train,[num2str(s_id(i)),'/',num2str(j),'/',im_list(k).name]);
            fprintf(f_train,' %f %f\n',s_attr(i,1), s_attr(i,2));
          
        end



    end
end
%%%%%%%%%%% organize testing data %%%%%%%%%%%%%%%
for i=1:n_v
    for j=1:modules % 5 views
        im_list = dir([testDir,'/',num2str(s_v_id(i)),'/',num2str(j),'/*.jpg']);
        im_n_v(i,j) = length(im_list);      
                
        for k=1:im_n_v(i,j)
          
            fprintf(f_test,[num2str(s_v_id(i)),'/',num2str(j),'/',im_list(k).name]);
            fprintf(f_test,' %f %f\n',s_v_attr(i,1), s_v_attr(i,2));
          
        end
    end
end

fclose(f_train);
fclose(f_test);

%max(im_n,[],1)
