#from __future__ import print_function
import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

from keras import backend as K

from keras.layers import concatenate

from sklearn.metrics import cohen_kappa_score

#import tensorflow as tf

import math
import random
from keras import optimizers
import numpy as np
import scipy.io as spio
#import sklearn
from sklearn.metrics import f1_score, accuracy_score


from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Layer,Dense, Dropout, Input, Activation, TimeDistributed, Reshape
from keras.layers import  GRU, Bidirectional
from keras.layers import  Conv1D, Conv2D, MaxPooling2D, Flatten, BatchNormalization, LSTM, ZeroPadding2D, GlobalAveragePooling2D
#from loadData import  load_data, classes_global, gen_data_seq, load_recording, gen_data
from keras.callbacks import History 
from keras.models import Model

from keras.layers.noise import GaussianNoise

from collections import Counter

from sklearn.utils import class_weight

def build_model(data_dim, n_channels, n_cl):
	eeg_channels = 1
	eog_channels = 2
	# here we create our convolution part
	def cnn_block(input_shape):
		act_conv = 'elu'
		init_conv = 'glorot_normal'
		dp_conv = 0.1
		input = Input(shape=input_shape)
		# 1st argument: number of filters
		# 2nd: size of the filter, 4 in x direction and 1 in y dimension, which effectively makes our convolution one dimensional
		# strides is the amount of samples to shift the filter, 1,1 is default 
		x = BatchNormalization()(input)
		x = Conv2D(32, (4, 1), strides=(1, 1), padding='same', kernel_initializer=init_conv)(x)
		# for sequential model you can substitute it with something like
		# model.add(Conv2D(args))
		# activation function, try LeakyReLu
		x = Activation(act_conv)(x)
		x = BatchNormalization()(x)
		x = Dropout(dp_conv)(x)
		# create 9 more layers, they are the same, but have more filters
		for i in range(9):
			x = Conv2D(64, (4, 1), strides=(1, 1), padding='same', kernel_initializer=init_conv)(x)
			x = Activation(act_conv)(x)
			x = BatchNormalization()(x)
			# MaxPooling reduces the dimention and makes 1 pixel out of 2
			x = MaxPooling2D(padding="same", pool_size=(2, 1))(x)
			x = Dropout(dp_conv)(x)
			# you can add some dropouts here, try the rate 0.25 for the beginning
		#here we put more filters with bigger size and stride
		# the goal is to have a tensor with a dimention 1x1xMany_filters
		x = Conv2D(128, (5, 1), strides=(5, 1), padding='same', kernel_initializer=init_conv)(x)
		x = Activation(act_conv)(x)
		x = BatchNormalization()(x)
		x = Dropout(dp_conv)(x)
		# flatten the output
		flatten1 = Flatten()(x)
		# creates the computational graph with corresponding input and output
		cnn_eeg = Model(inputs=input, outputs=flatten1)
		return cnn_eeg
		

	# parameters for lstm; size and dropout rate
	hidden_units  = 32
	dp = 0.4


	# create a cnn block for every part 
	input_eeg = Input(shape=( None, data_dim, 1, eeg_channels))
	cnn_eeg = cnn_block(( data_dim, 1, eeg_channels))
	# time distributed makes sure that the weights are shared across the timestep
	x_eeg = TimeDistributed(cnn_eeg)(input_eeg)


	if n_channels>1:
		input_eog = Input(shape=( None, data_dim, 1, eog_channels))
		cnn_eog = cnn_block(( data_dim, 1, eog_channels))
		x_eog = TimeDistributed(cnn_eog)(input_eog)



	if n_channels==3:
		input_emg = Input(shape=(None, 1), name='emg_input', dtype='float32')

	if n_channels==1:
		x = x_eeg
	elif n_channels==2:
		x = concatenate([ x_eeg, x_eog])
	else:
		x = concatenate([ x_eeg, x_eog, input_emg ])


	# LSTM layers
	x = BatchNormalization()(x)
	x = Bidirectional(LSTM(units=hidden_units,
				   return_sequences=True, activation='tanh',
				   recurrent_activation='sigmoid', dropout = dp, recurrent_dropout = dp))(x)
	x = BatchNormalization()(x)
	x = Bidirectional(LSTM(units=hidden_units,
				   return_sequences=True, activation='tanh',
				   recurrent_activation='sigmoid', dropout = dp, recurrent_dropout = dp))(x)
	x = BatchNormalization()(x)
	predictions = TimeDistributed(Dense(units=n_cl, activation='softmax'))(x)

	# create a model with all the inputs and one output

	if n_channels==1:
		model = Model(inputs=[input_eeg] , outputs=[predictions])
	elif n_channels==2:
		model = Model(inputs=[input_eeg, input_eog] , outputs=[predictions])
	else:
		model = Model(inputs=[input_eeg, input_eog, input_emg] , outputs=[predictions])


	return [cnn_eeg, model]