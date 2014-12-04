clear,close all,clc

rootfd ='~/work/data/classification_resize/';
rootfd_test = '~/work/data/verification_resize/';
listfd ='~/work/caffe/caffe-mmlab-mmlab_shared_buffer/data/car/';
dbfd='~/work/caffe_data/';
attr_type='attr_cont';
iters=ones(5,1)*2000;
layers = {'fc8_speed','fc8_disl'};

%prep test data

if exist([dbfd,'test_leveldb'],'dir')
    rmdir([dbfd,'test_leveldb'],'s');
end

cmd =['convert_imageset.bin ',rootfd_test,' ',listfd,'test_car_',attr_type,...
   ' ',dbfd,'test_leveldb 0'];

cmd
system(cmd);

model_name = ['imagenet_finetune_car_',attr_type,'_p2_iter_',num2str(1000)];
model_proto_file = ['../../examples/imagenet_ft_car/imagenet_finetune_overfeat_',...
    attr_type,'_test.prototxt'];
for layer = 1:length(layers)
    extract_layer_name = layers{layer};
    save_name = ['test_prob_car_',extract_layer_name];
    if exist(save_name,'dir')
        rmdir(save_name,'s');
    end

    minibatch_n = '35';%adjust to fit the real datasize
    cmd = ['extract_features.bin ',model_name,' ',model_proto_file,' ',...
        extract_layer_name,' ',save_name,' ',minibatch_n, ' GPU'];

    system(cmd);
end
