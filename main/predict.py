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

n_channels = 3
# folder with mat files of the data you want to score
data_dir = './../mat/'
# folder for the output, you will get a mat file with stages and spectrogram
out_dir = './score_sleep/pred/'

# parameters of the model, we suggest to use EEG, EOG and EMG
seq_length = 8



ordering = 'tf';
K.set_image_dim_ordering(ordering)

# build our model
[cnn_eeg, model] = build_model(data_dim, n_channels, n_cl)

# load model, the one below was the best one for our data
model.load_weights('./model_h_3ch.h5')


# this is an array with the scoring
y_ = []
# this is ground truth, in case you just scre data it will be set to zeros
y = []
# array with predicted probabilities of the classes
y_p = []



files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
f_list = files
for i in range(0,len(f_list)):
	f = f_list[i]
	X1, X2,  targets, emg,Pspec = load_recording(data_dir, f )
	X1 = np.expand_dims( X1, 0)
	X2 = np.expand_dims( X2, 0)
	emg = np.expand_dims( emg, 0)
	targets = np.expand_dims( targets, 0)
	if n_channels == 1:
		sample = [X1]
	elif n_channels == 2:
		sample = [X1, X2]
	else:
		sample = [X1, X2, emg]
	sample_y = targets
	sample_len = X1.shape[1]
 	#scoring 
	y_pred = model.predict( sample , batch_size=1, verbose=1)
	y_ = np.argmax(y_pred, axis=2).flatten() 
	y = np.argmax(targets, axis=2).flatten() 
	y_p = y_pred
 	#save the output 
	scipy.io.savemat( out_dir+f+'.mat', mdict={ 'y_p':y_p,  'y_': y_, 'y':y, 'Pspec':Pspec})



	
	
