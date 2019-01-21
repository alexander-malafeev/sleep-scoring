import scipy.io as spio
from os import listdir
from os.path import isfile, join
import re
import numpy as np
import math
from random import randint
from keras.utils import np_utils

def load_recording(dir_name, f_name):

	# load mat file with data
	mat = spio.loadmat(dir_name+f_name, struct_as_record=False, squeeze_me=True)
	stages = mat['Data'].stages
	Pspec =  mat['Data'].Pspec
 
	# take EEG channel C3A2, ocular channels and value of EMG power
	C3A2 = mat['Data'].C3A2
	LOC = mat['Data'].LOC
	ROC =  mat['Data'].ROC
	PEMG =  mat['Data'].PEMG

	# transpose matrices 
	C3A2 = np.transpose(C3A2)
	LOC = np.transpose(LOC)
	ROC = np.transpose(ROC)
	PEMG = np.transpose(PEMG)
	
	# add dimensions
	PEMG =  np.expand_dims( PEMG, 1)	
	C3A2 =  np.expand_dims( C3A2, 3)
	C3A2 =  np.expand_dims( C3A2, 4)
	LOC =  np.expand_dims( LOC, 3)
	LOC =  np.expand_dims( LOC, 4)
	ROC =  np.expand_dims( ROC, 3)
	ROC =  np.expand_dims( ROC, 4)
	
	# concatenate ocular channels
	EOG = np.concatenate( ( LOC, ROC ), axis = 3 )

	# transform targets to vector format
	targets = np_utils.to_categorical(stages, 5)
	
	# we scale and clip the data for better training
	C3A2 = (C3A2 )/150 
	C3A2 = np.clip(C3A2, -1, 1)
	
	EOG = (EOG )/150 
	EOG = np.clip(EOG, -1, 1)
	
	PEMG = np.clip(PEMG, -1, 1)
	
	return (C3A2, EOG, targets, PEMG,Pspec )


def get_classes( dir_name, f_name ):
	# this function just takes the stages vector from, we need it to compute the weights of the classes
	mat = spio.loadmat(dir_name+f_name, struct_as_record=False, squeeze_me=True)
	stages = mat['Data'].stages
	st = stages;
	return (st)


def get_classes_global(data_dir, files_train):
	# this function computes the amount of epochs of every class in the whole dataset using the previous function get_classes
	print("=====================")
	print("Reading train set:")
	f_list = files_train
	train_l = []
	st0 = []
	for i in range(0,len(f_list)):
		f = f_list[i]
		st  = get_classes(data_dir, f )
		st0.extend(st)
	return st0



def gen_data_seq(data_dir, files_train,  seq_length, n_samples,  sample_files, data_dim, n_classes=5 ):
	# we need this function to read data for the generator
	# we read just few files. Database can be very large and we often can not load it to the memory
	# sample_files is the number of files to be read at once
	print('====Generator====')
	# choose sample_files randomly
	f_list = np.random.choice(files_train,sample_files, replace = False )
	
	# number of training sequences to be generated; we take "sample" sequences from each file
	n_seq_train =  len(f_list)*n_samples 
	
	# create empty matrices for 
	# EEG
	data_train1 = np.zeros(( n_seq_train, seq_length, data_dim, 1, 1 ))
	# EOG
	data_train2 = np.zeros(( n_seq_train, seq_length, data_dim, 1, 2 ))
	# and EMG data; dimensions are following: samples, epochs, features_X, features_Y, channels
	data_EMG = np.zeros(( n_seq_train, seq_length,  1 ))
	# and for ground truth stages
	targets_train =  np.zeros(( n_seq_train, seq_length, n_classes ))


	print("=====================")
	print("Reading training set:")
	j = 0
	for i in range(0,len(f_list)):
		f = f_list[i]
		print(f)
		# load data from file, X1 - EEG, X2 - EOG
		X1, X2, targets, PEMG, _ = load_recording(data_dir, f )
		# number of the epochs in the recording
		l= targets.shape[0]
		
		# get n_samples sequences from a recording, each sequence contains seq_length epochs
		for k in range(0,n_samples):
			# choose the start of the sequence randomly, but it should be within the length of the recording
			seq_start = randint(0,l-seq_length)
			# add sampled sequence into the resulting array
			# k+j is the current index
			data_train1[j] = X1[seq_start:seq_start+seq_length,:]
			data_train2[j] = X2[seq_start:seq_start+seq_length,:]
			data_EMG[j] = PEMG[seq_start:seq_start+seq_length,:]
			targets_train[j] = targets[seq_start:seq_start+seq_length,:]
			j = j+1	
	
	return ( data_train1,data_train2, targets_train, data_EMG )


