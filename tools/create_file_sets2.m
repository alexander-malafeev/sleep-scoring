clear; clc; close all; fclose all;
rng(0); % random number generator seed for reproducibility

% this script splits training data set into training, cross validation and
% testing datasets in proportion 70:15:15. Every recording in our dataset
% starts with letters VP, then it is followed by an integer subject 
% identifier. Dataset can contain several recordings of the same subject.
% we want to put all recordings of the same subject into the same subset.
% this script performs such a split. If it is not a concern you can use the
% script create_file_sets1.m. If you use this script you should make sure
% subject identifier in the code is the same as in your datafolder.

% ==split recordings per subject to avoid one subject in training 
% and cv sets
subjects = {};
files_per_subj = {};
num_files_per_subj = [];
for i=1:numel(f_list)
    str = f_list(i).name;
    expression = ['VP\d{2}'];
    VP = regexp(str,expression,'match');
    idx = find(strcmp([subjects{:}], VP));
    if numel(idx) == 0
        idx = numel(subjects)+1;
        subjects(idx) = {VP};
        num_files_per_subj(idx) = 0;
    end
    num_files_per_subj(idx) = num_files_per_subj(idx) +1;
    files_per_subj{idx,num_files_per_subj(idx)} = f_list(i).name;
end


IDX = randperm( numel(subjects) );



N =  numel(subjects);
N_train = round( pt*N );
N_CV = round( pCV*N );
N_test = N - N_train - N_CV; 

ndx_subj_train = IDX(1:N_train);
ndx_subj_CV = IDX(N_train+1:N_train+N_CV);
ndx_subj_test = IDX(N_train+N_CV+1:end);

subj_train = subjects(ndx_subj_train);
subj_CV = subjects(ndx_subj_CV);
subj_test = subjects(ndx_subj_test);
files_train = {};
for i=1:numel(ndx_subj_train)
    j = ndx_subj_train(i);
    files_train = [ files_train; reshape(files_per_subj(j,1:num_files_per_subj(j)),[],1);]
end


files_CV= {};
for i=1:numel(ndx_subj_CV)
    j = ndx_subj_CV(i);
    files_CV = [ files_CV; reshape(files_per_subj(j,1:num_files_per_subj(j)),[],1);]
end

files_test= {};
for i=1:numel(ndx_subj_test)
    j = ndx_subj_test(i);
    files_test = [ files_test; reshape(files_per_subj(j,1:num_files_per_subj(j)),[],1);]
end

fprintf('number of subjects %s \n', num2str(N));
fprintf('subjects in Training set %s \n', num2str(numel(subj_train)));
fprintf('in CV set %s \n', num2str(numel(subj_CV)));
fprintf('in Test set %s \n', num2str(numel(subj_test)));

fprintf('=================================\n');

fprintf('number of files %s \n', num2str(numel(f_list)));
fprintf('files in Training set %s \n', num2str(numel(files_train)));
fprintf('in CV set %s \n', num2str(numel(files_CV)));
fprintf('in Test set %s \n', num2str(numel(files_test)));

save('file_sets.mat', 'files_train', 'files_CV', 'files_test');