#from __future__ import print_function
import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

from keras import backend as K

from keras.layers import concatenate


from keras import optimizers

import numpy as np
import scipy.io 
#import sklearn
from sklearn.metrics import f1_score, accuracy_score

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Layer,Dense, Dropout, Input, Activation, TimeDistributed, Reshape, Bidirectional
from keras.layers import  GRU
from keras.layers import Convolution1D, MaxPooling1D,  Conv2D, MaxPooling2D, Flatten, BatchNormalization, LSTM, ZeroPadding2D #,Merge 
from loadData import   load_recording
from keras.callbacks import History 
from keras.models import Model

from keras.layers.noise import GaussianNoise
from loadData import load_recording

from os import listdir
from os.path import isfile, join

from myModel import build_model

fs  = 128;
data_dim = fs*20

n_cl = 5


# folder with mat files of the data you want to score
data_dir = '/home/alex/Pharma/data/scoring/Sebastian/'
# folder for the output, you will get a mat file with stages and spectrogram
out_dir = './score_sleep/pred/'

# parameters of the model, we suggest to use EEG, EOG and EMG
seq_length = 8

n_channels = 3

ordering = 'tf';
K.set_image_dim_ordering(ordering)

# build our model
[cnn_eeg, model] = build_model(data_dim, n_channels, n_cl)

# load model, the one below was the best one for our data
model.load_weights('./model_h_3ch.h5')







# since we can not run LSTM on too long sequences due to memory limitations
# lets split the recording
segment_length = 1000

files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
f_list = files
for i in range(0,len(f_list)):
	f = f_list[i]
	print(f)
	X1, X2,  targets, emg,Pspec = load_recording(data_dir, f )
	X1 = np.expand_dims( X1, 0)
	#print(X1.shape)
	X2 = np.expand_dims( X2, 0)
	emg = np.expand_dims( emg, 0)
	
	targets = np.expand_dims( targets, 0)
	recording_length = X1.shape[1]
	# this is an array with the scoring
	y_ = np.zeros(recording_length)
	# this is ground truth, in case you just scre data it will be set to zeros
	y = np.zeros(recording_length)
	# array with predicted probabilities of the classes
	y_p = np.zeros((recording_length,5))
	i = 0
	while i<recording_length:
		i_new = i + segment_length
		i_new = min(i_new, recording_length)
		
		X1_batch = X1[:, i:i_new, :, :, :]
		X2_batch = X2[:, i:i_new, :, :, :]
		emg_batch = emg[:, i:i_new, :]
		targets_batch = targets[:,i:i_new, :]
		
		if n_channels == 1:
			sample = [X1_batch]
		elif n_channels == 2:
			sample = [X1_batch, X2_batch]
		else:
			sample = [X1_batch, X2_batch, emg_batch]
		sample_y = targets
		#sample_len = X1.shape[1]
		#scoring 
		y_pred = model.predict( sample , batch_size=1, verbose=1)
		y_[i:i_new] = np.argmax(y_pred, axis=2).flatten() 
		y[i:i_new] = np.argmax(targets_batch, axis=2).flatten() 
		y_p[i:i_new] = y_pred
			
		i = i_new
	
	
	
 	#save the output 
	scipy.io.savemat( out_dir+f+'.mat', mdict={ 'y_p':y_p,  'y_': y_, 'y':y, 'Pspec':Pspec})



	
	
