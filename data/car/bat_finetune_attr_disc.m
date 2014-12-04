clear,close all,clc

rootfd ='~/work/data/classification_resize/';
rootfd_test = '~/work/data/verification_resize/';
listfd ='~/work/caffe/caffe-mmlab-mmlab_shared_buffer/data/car/';
dbfd='~/work/caffe_data/';
attr_type='attr_disc';

for i=2:5
    i
    %prep data
    if exist([dbfd,'train_leveldb'],'dir')
        rmdir([dbfd,'train_leveldb'],'s');
    end
    if exist([dbfd,'test_leveldb'],'dir')
        rmdir([dbfd,'test_leveldb'],'s');
    end
    
    cmd= ['convert_imageset.bin ',rootfd,' ',listfd,'train_car_',attr_type,'_',num2str(i),...
        ' ',dbfd,'train_leveldb 1'];
    cmd
    system(cmd);
    cmd =['convert_imageset.bin ',rootfd_test,' ',listfd,'test_car_',attr_type,'_',num2str(i),...
       ' ',dbfd,'test_leveldb 0'];
    system(cmd);
    %cmd =['convert_imageset.bin ',rootfd,' ',listfd,'test_part_',num2str(i),...
     %   ' ',dbfd,'test_part_leveldb 0'];

    %write solver
    solv_name = '../../examples/imagenet_ft_car/imagenet_finetune_overfeat_solver.bak';
    solv_new = '../../examples/imagenet_ft_car/imagenet_finetune_overfeat_solver.prototxt';
    f1=fopen(solv_name,'r');
    f2=fopen(solv_new,'w');

    fprintf(f2,['train_net: \"/home/ljyang/work/caffe/caffe-mmlab-mmlab_shared_buffer/',...
        'examples/imagenet_ft_car/imagenet_finetune_overfeat_',attr_type,'_train.prototxt\"\n']);
    fprintf(f2,['test_net: \"/home/ljyang/work/caffe/caffe-mmlab-mmlab_shared_buffer/',...
        'examples/imagenet_ft_car/imagenet_finetune_overfeat_',attr_type,'_test.prototxt\"\n']);
    fprintf(f2,'test_iter: 2\n');
    fprintf(f2,'test_interval: 50\n');
    fprintf(f2,'base_lr: 0.001\n');
    for k=1:5
        line = fgetl(f1);
    end
    for k=6:7
        line = fgetl(f1);
        fprintf(f2,'%s\n',line);
    end
    

    fprintf(f2,'stepsize: 1000\n');
    fprintf(f2,'display: 20\n');
    fprintf(f2,'max_iter: 2000\n');
    fprintf(f2,'momentum: 0.9\n');
    fprintf(f2,'weight_decay:0.0005\n');
    fprintf(f2,'snapshot: 1000\n');
    %L2 loss param
    fprintf(f2,'test_compute_loss: true\n');
    
    fprintf(f2,'snapshot_prefix: ');
    fprintf(f2, '\"imagenet_finetune_car_%s_%d\"\n',attr_type,i);
    fprintf(f2, 'solver_mode: GPU\ndevice_id:0\n');
    fclose(f1);
    fclose(f2);
    %train model: fine-tune from car model classification model is not good
%     cmd=['GLOG_logtostderr=1 finetune_net.bin ',...
%         '../../examples/imagenet_ft_car/imagenet_finetune_overfeat_solver.prototxt ',...
%         'imagenet_finetune_car_',attr_type,'_',num2str(i),'_iter_1400'];

    cmd=['GLOG_logtostderr=1 finetune_net.bin ',...
        '../../examples/imagenet_ft_car/imagenet_finetune_overfeat_solver.prototxt ',...
        '../../examples/imagenet_ft_car/imagenet-overfeat_iter_860000'];
    system(cmd);

end
