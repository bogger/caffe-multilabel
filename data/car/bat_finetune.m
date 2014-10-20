clear,close all,clc

rootfd ='~/work/data/classification_part_resize/';
rootfd_test = '~/work/data/classification_part_test_crop/';
listfd ='~/work/caffe/caffe-mmlab-mmlab_shared_buffer/data/car/';
dbfd='~/work/caffe_data/';
for i=1:8
    %prep data
    if exist([dbfd,'train_leveldb'],'dir')
        rmdir([dbfd,'train_leveldb'],'s');
    end
    if exist([dbfd,'test_leveldb'],'dir')
        rmdir([dbfd,'test_leveldb'],'s');
    end
    cmd= ['convert_imageset.bin ',rootfd,' ',listfd,'train_part_',num2str(i),...
        ' ',dbfd,'train_leveldb 1'];
    cmd
    system(cmd);
    cmd =['convert_imageset.bin ',rootfd_test,' ',listfd,'test_part_crop_',num2str(i),...
       ' ',dbfd,'test_leveldb 0'];
    
    %cmd =['convert_imageset.bin ',rootfd,' ',listfd,'test_part_',num2str(i),...
     %   ' ',dbfd,'test_part_leveldb 0'];
    
    
    cmd
    system(cmd);
    %compute mean
%     cmd = ['compute_image_mean.bin ',dbfd,'train_part_leveldb car_part_mean.binaryproto'];
%     system(cmd);
%     cmd=['compute_image_mean.bin ',dbfd,'test_part_leveldb car_part_crop_mean.binaryproto'];
%     system(cmd);
    %write solver
    solv_name = '../../examples/imagenet_ft_car/imagenet_finetune_overfeat_solver.bak';
    solv_new = '../../examples/imagenet_ft_car/imagenet_finetune_overfeat_solver.prototxt';
    f1=fopen(solv_name,'r');
    f2=fopen(solv_new,'w');
    for k=1:13
        line = fgetl(f1);
        fprintf(f2,'%s\n',line);
    end
    fprintf(f2,'snapshot_prefix: ');
    fprintf(f2, '\"imagenet_finetune_part_%d\"\n',i);
    fprintf(f2, 'solver_mode: GPU\ndevice_id:0\n');
    fclose(f1);
    fclose(f2);
    %train model
    cmd=['GLOG_logtostderr=1 finetune_net.bin ',...
        '../../examples/imagenet_ft_car/imagenet_finetune_overfeat_solver.prototxt ',...
        '../../examples/imagenet_ft_car/imagenet-overfeat_iter_860000'];
    system(cmd);
end
