clear,close all,clc

rootfd ='~/work/data/classification_part_resize/';
rootfd_test = '~/work/data/classification_part_test_crop/';
listfd ='~/work/caffe/caffe-mmlab-mmlab_shared_buffer/data/car/';
dbfd='~/work/caffe_data/';
for i=1:8
    %prep test data

    if exist([dbfd,'test_part_leveldb'],'dir')
        rmdir([dbfd,'test_part_leveldb'],'s');
    end

    cmd =['convert_imageset.bin ',rootfd_test,' ',listfd,'test_part_crop_',num2str(i),...
       ' ',dbfd,'test_part_leveldb 0'];

    cmd
    system(cmd);
    
    model_name = ['imagenet_finetune_part_',num2str(i),'_iter_900'];
    model_proto_file = '../../examples/imagenet_ft_car/imagenet_finetune_overfeat_test.prototxt';
    extract_layer_name = 'prob_ft';
    save_name = ['test_prob_part_',num2str(i)];
    minibatch_n = '17';
    cmd = ['extract_features.bin ',model_name,' ',model_proto_file,' ',...
        extract_layer_name,' ',save_name,' ',minibatch_n, ' GPU'];
    
    system(cmd);
end