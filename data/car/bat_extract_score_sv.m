clear,close all,clc

rootfd ='~/work/data/surveillance2_resize/';%modified

listfd ='~/work/caffe/caffe-mmlab-mmlab_shared_buffer/data/car/';
dbfd='~/work/caffe_data/';

    %prep test data

    if exist([dbfd,'test_leveldb'],'dir')
        rmdir([dbfd,'test_leveldb'],'s');
    end

    cmd =['convert_imageset.bin ',rootfd,' ',listfd,'surveillance_car',...
       ' ',dbfd,'test_leveldb 0'];

    cmd
    system(cmd);
    
    model_name = ['imagenet_finetune_car_iter_4000'];
    model_proto_file = '../../examples/imagenet_ft_car/imagenet_finetune_overfeat_test_center_crop.prototxt';
    extract_layer_name = 'prob_ft';
    save_name = ['sv_car'];
    if exist(save_name,'dir')
        rmdir(save_name,'s');
    end
    
    minibatch_n = '1';
    cmd = ['extract_features.bin ',model_name,' ',model_proto_file,' ',...
        extract_layer_name,' ',save_name,' ',minibatch_n, ' GPU'];
    
    system(cmd);