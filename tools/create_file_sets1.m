clear; clc; close all; fclose all;

rng(0);

train_p = 0.7;
val_p = 0.15;
test_p = 1-train_p-val_p;



dataFolder = './data/';
fileList = dir( [ dataFolder '*.mat' ] );

n = numel(fileList);



ndx = randperm(n);

train_ndx = ndx(1:round(n*train_p));
val_ndx = ndx(numel(train_ndx)+1:round(n*(train_p+val_p)));
test_ndx = ndx(numel(train_ndx)+numel(val_ndx)+1:end);

files_train = {}
files_val = {}
files_test = {}

for i=1:numel(train_ndx)
    files_train{i} = fileList(train_ndx(i)).name;
end


for i=1:numel(val_ndx)
    files_val{i} = fileList(val_ndx(i)).name;
end

for i=1:numel(test_ndx)
    files_test{i} = fileList(test_ndx(i)).name;
end

files_train = files_train';
files_val = files_val';
files_test = files_test';

save('file_sets_02.mat','files_train', 'files_val', 'files_test');