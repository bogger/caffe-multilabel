clear,close all,clc
type ='verification_add';
datDir=['~/work/data/',type];
savDir =['~/work/data/',type,'_resize'];
load select_data_add%s_id, s_v_id

rng(0);%rand seed
%comment this line for classification data
n=length(v_a_id);
im_n_a = zeros(n,1);

for i=1:n
    
        srcDir = [datDir,'/',num2str(v_a_id(i))];
        desDir = [savDir,'/',num2str(v_a_id(i))];
        if ~exist(desDir,'dir')
            mkdir(desDir);
        end
        im_list = dir([srcDir,'/*.jpg']);
        im_n_a(i) = length(im_list);
        for k=1:im_n_a(i)
            im = imread([srcDir,'/',im_list(k).name]);
            im =imresize(im, [256,256]);
            imwrite(im,[desDir,'/',im_list(k).name]);
        end
   
end

save im_n_a im_n_a

