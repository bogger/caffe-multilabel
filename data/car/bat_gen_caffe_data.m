
rootfd ='~/work/data/classification_part_resize/';
rootfd_test = '~/work/data/classification_part_test_crop/';
listfd ='~/work/caffe/caffe-mmlab-mmlab_shared_buffer/data/car/';
dbfd='~/work/caffe_data/';
for i=1:1
    cmd= ['convert_imageset.bin ',rootfd,' ',listfd,'train_part_',num2str(i),...
        ' ',dbfd,'train_part_',num2str(i),'_leveldb 1'];
    cmd
    system(cmd);
    cmd =['convert_imageset.bin ',rootfd_test,' ',listfd,'test_part_',num2str(i),...
        ' ',dbfd,'test_part_',num2str(i),'_leveldb 0'];
    cmd
    system(cmd);
end
