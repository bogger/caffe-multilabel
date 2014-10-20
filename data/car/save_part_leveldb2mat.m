clear,close all,clc
for i=1:1
    level_db_name = ['test_prob_part_',num2str(i)];
    cmd =['python leveldb2mat.py ',level_db_name,' 4420 442 ',level_db_name,'.mat'];
    system(cmd);
    %res = evalc(['system(''',cmd,''')']);
end

