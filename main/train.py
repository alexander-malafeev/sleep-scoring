#from __future__ import print_function
import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

from keras import backend as K
from keras.layers import concatenate

from keras import optimizers

import numpy as np
import scipy.io as spio
from sklearn.metrics import f1_score, accuracy_score
np.random.seed(0)  

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Layer,Dense, Dropout, Input, Activation, TimeDistributed, Reshape, Bidirectional
from keras.layers import  GRU
from keras.layers import Convolution1D, MaxPooling1D,  Conv2D, MaxPooling2D, Flatten, BatchNormalization, LSTM, ZeroPadding2D
from loadData import  get_classes_global, gen_data_seq, load_recording
from keras.callbacks import History 
from keras.models import Model

from keras.layers.noise import GaussianNoise

from collections import Counter

from sklearn.utils import class_weight
from myModel import build_model

# sampling rate, downsampled to reduce size of the network
fs  = 128;
# length of the epoch in samples
data_dim = fs*20

# number of classes; 0-wake, 1-3, 4-REM
n_cl = 5


data_dir = './../../../data/scoring/data128/'

# contains list of recordings for training, validation and test (70%, 15%, 15%)
f_set = './../../../data/scoring/file_sets_00.mat'

mat = spio.loadmat(f_set)
files_train = []

files_test = []
	
files_CV = []

# create python lists out of matlab arrays	
tmp =  mat['files_train']
#tmp =  mat['train_files']
for i in range(len(tmp)):
	file = [str(''.join(l)) for la in tmp[i] for l in la]
	files_train.extend(file)
	
tmp =  mat['files_CV']
#tmp =  mat['test_files']
for i in range(len(tmp)):
	file = [str(''.join(l)) for la in tmp[i] for l in la]
	files_CV.extend(file)
	
tmp =  mat['files_test']
#tmp =  mat['test_files']

for i in range(len(tmp)):
	file = [str(''.join(l)) for la in tmp[i] for l in la]
	files_test.extend(file)
	
# length of the sequence
seq_length = 8
	
# sample in our case is a sequence of epochs, each seq_length epochs long
samples_per_batch = 64

#batch  generator loads part of the training set into memory, namely sample_files
sample_files = 16
# and samples "samples" number of sequences from each file and forms batches, each samples_per_batch samples
samples = 100


# here we store length of each recording
lengths = []

# total number of channels to use (only one EEG)
n_channels = 3
eeg_channels = 1
eog_channels = 2

st0 = get_classes_global(data_dir, files_train)
cls = np.arange(n_cl)

cl_w = class_weight.compute_class_weight('balanced', cls, st0)

print(cl_w)




def myGenerator():
	#global stages
	# X1 - data matrix of EEG
	# X2 - data matrix of EOG (2 channels)
	# X_emg is an array with corresponding power of EMG (1 value per epoch)
	# y_train - labels
	(X1, X2, y_train, X_emg) = gen_data_seq(data_dir, files_train,  seq_length, samples, sample_files, data_dim, n_cl )
	# shuffle the matrix
	ndx = np.random.permutation(X1.shape[0])
	np.take(X1,ndx,axis=0,out=X1)
	np.take(X2,ndx,axis=0,out=X2)
	np.take(y_train,ndx,axis=0,out=y_train)
	np.take(X_emg,ndx,axis=0,out=X_emg)
	# generator infinite loop
	while 1:
		# form next batch
		for i in range((sample_files*samples)//samples_per_batch):
			xx1 = X1[i*samples_per_batch:(i+1)*samples_per_batch]
			xx2 = X2[i*samples_per_batch:(i+1)*samples_per_batch]
			emg = X_emg[i*samples_per_batch:(i+1)*samples_per_batch]
			y = y_train[i*samples_per_batch:(i+1)*samples_per_batch]
			
			# weights array
			s_w = np.zeros((y.shape[0], y.shape[1]))
			# I compute wieghts for every batch, but you can compute it once for the whole
			# training dataset, it would give the same result
			y_int = np.argmax(y, axis=2).flatten();
			#cl_w = get_class_weights(y_int)
			# fill the weight matrix
			for i in range(y.shape[0]):
				for j in range(y.shape[1]):
					s_w[i, j] = cl_w[np.argmax(y[i,j,:])]
			# return the batch
			if n_channels == 1:
				yield  [xx1],  y, s_w
			elif n_channels == 2:
				yield  [xx1, xx2],  y, s_w
			else:
				yield  [xx1, xx2, emg],  y, s_w


# sets the ordering of tensor dimentions in the format of tensorflow  [samples, W, H, Channels]
ordering = 'tf';
K.set_image_dim_ordering(ordering)


# build our model
[cnn_eeg, model] = build_model(data_dim, n_channels, n_cl)

# create an optimizer
opt = optimizers.RMSprop( clipnorm=1. )
model.compile(optimizer=opt,  loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode="temporal")

# print model
print(cnn_eeg.summary())
print(model.summary())



#===========================================================
# training and validation

print(model.metrics_names)

history = History()

n_ep = 50
F1_cv_tmp = []
F1_tst_tmp = []
F1_cv = np.zeros( (n_ep, 5) )
F1_tst = np.zeros( (n_ep, 5) )

acc_cv = []
acc_tst = []
acc_tr = []
loss_tr = []
loss_tst = []
loss_cv = []

def validate_model(model, fileset, n_chan =1):
	y_ = []
	y = []
	loss_tmp = []
	f_list = fileset
	CV_l = []
	for j in range(0,len(f_list)):
		f = f_list[j]
		X1, X2,  targets, emg, _  = load_recording(data_dir, f )
	
		CV_l.append(X1.shape[0])
		X1 = np.expand_dims( X1, 0)
		X2 = np.expand_dims( X2, 0)
		emg = np.expand_dims( emg, 0)
		targets = np.expand_dims( targets, 0)
		if n_chan == 1:
			sample = [X1]
		elif n_chan == 2:
			sample = [X1, X2]
		else:
			sample = [X1, X2, emg]
			
		sample_y = targets
		
		y_pred = model.predict(  sample, batch_size=X1.shape[0], verbose=1)
		scores = model.evaluate(sample, sample_y, batch_size=X1.shape[0])
		loss_tmp.append(scores[0])
		y_.extend( np.argmax(y_pred, axis=2).flatten() )
		y.extend( np.argmax(sample_y, axis=2).flatten() )
	f1 = f1_score( y, y_, average=None )
	acc = accuracy_score(y, y_)
	loss = np.mean(loss_tmp)
	return f1, acc, loss, y, y_, CV_l

for i in range(n_ep):
	print("Epoch = " + str(i))
	my_generator = myGenerator()
	model.fit_generator(my_generator, steps_per_epoch = (sample_files*samples)//samples_per_batch, epochs = 1, verbose=1,  callbacks=[history], initial_epoch=0)
	acc_tr.append(history.history['acc'])
	loss_tr.append(history.history['loss'])
	
	f1_1, acc_1, loss_1, cv_y, cv_y_, CV_l = validate_model(model, files_CV, n_channels)
	
	loss_cv.append(loss_1)
	#print(loss_cv)
	F1_cv_tmp = f1_1
	F1_cv[i,:] = F1_cv_tmp
	#print(F1_cv.shape)
	acc_cv.append(acc_1)
	print( "epoch = ", i )
	print( "F1 cv = ", f1_1 )
	print( "acc cv = ", acc_1 )
	print( "loss cv = ", loss_cv[-1] )
	

	f1_2, acc_2, loss_2, t_y, t_y_, test_l = validate_model(model, files_test, n_channels)
	
	loss_tst.append(loss_2)

	F1_tst[i,:] = f1_2
	acc_tst.append(acc_2)
	print( "F1 test = ", f1_2)
	print( "acc test = ", acc_2 )
	print( "loss test = ", loss_tst[-1] )
	
	# save the model after each training iteration, this way we can choose the best one
	model.save('./models/model_ep'+str(i)+'.h5')
	spio.savemat('./predictions/predictions_ep'+str(i)+'.mat', mdict={'cv_y': cv_y, 'cv_y_': cv_y_, 't_y': t_y, 't_y_': t_y_, 'test_l': test_l, 'CV_l': CV_l, 'files_test':files_test, 'files_CV':files_CV, 'F1_tst':f1_2, 'F1_cv':f1_1  })



	

#model.load_weights('./model.h5')
#model.save('./model.h5')

spio.savemat('./predictions.mat', mdict={'cv_y': cv_y, 'cv_y_': cv_y_, 't_y': t_y, 't_y_': t_y_, 'test_l': test_l, 'CV_l': CV_l, 'seq_length':seq_length, 'files_test':files_test, 'files_CV':files_CV, 'acc_tr': acc_tr, 'loss_tr':loss_tr,'loss_tst':loss_tst, 'loss_cv':loss_cv, 'acc_tst':acc_tst, 'acc_cv':acc_cv, 'F1_tst':F1_tst, 'F1_cv':F1_cv  })
