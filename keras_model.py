# Usage: python keras_model.py <path_to_input> <path_to_output> <mode>
# Input: path_to_input: path to the directory where extracted_feat and labels are stored
# 	 path_to_output: path where train, val loss, test prediction for each model will be stored
#	 mode: test or train
# Output: Store arrays for the train and validation loss at each iteration along with the actual versus predicted for validation images

# Import all dependencies
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
import sys
import tables
import numpy as np
from scipy.stats import norm
import time


def root_mean_squared_error(y_true, y_pred):
    """
    RMSE loss function
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)


def get_model(model_type):
    model = Sequential()
    model.add(Activation(activation=center_normalize, input_shape=(512,28,28)))
   
    if model_type == 'model1' or model_type == 'model2' or model_type == 'model3':
	    model.add(Convolution2D(64, 3, 3, border_mode='same'))
	    model.add(Activation('relu'))
	    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
	    model.add(Activation('relu'))
	    model.add(ZeroPadding2D(padding=(1, 1)))
	    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	    model.add(Dropout(0.25))

    if model_type == 'model2' or model_type == 'model3':
	    model.add(Convolution2D(96, 3, 3, border_mode='same'))
	    model.add(Activation('relu'))
	    model.add(Convolution2D(96, 3, 3, border_mode='valid'))
	    model.add(Activation('relu'))
	    model.add(ZeroPadding2D(padding=(1, 1)))
	    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	    model.add(Dropout(0.25))
 
    if model_type == 'model3':
	    model.add(Convolution2D(128, 2, 2, border_mode='same'))
	    model.add(Activation('relu'))
	    model.add(Convolution2D(128, 2, 2, border_mode='same'))
	    model.add(Activation('relu'))
	    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, W_regularizer=l2(1e-3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss=root_mean_squared_error)
    return model

def load_data(path_in):
	
	f = tables.open_file(path_in+'extracted_feat/diastoleX.h5', mode='r')
	diastoleX = np.array(f.root.data)
	f.close()
	
	f = tables.open_file(path_in+'labels/diastoleY.h5', mode='r')
        diastoleY = np.array(f.root.data).T
        f.close()
	
	f = tables.open_file(path_in+'extracted_feat/systoleX.h5', mode='r')
	systoleX = np.array(f.root.data)
	f.close()

	f = tables.open_file(path_in+'labels/systoleY.h5', mode='r')
        systoleY = np.array(f.root.data).T
        f.close()
	
	return diastoleX, diastoleY, systoleX, systoleY

def split_data(X, y, split_ratio=0.2):
    """
    Split data into training and testing.
    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split = X.shape[0] * split_ratio
    X_test = X[:split, :, :, :]
    y_test = y[:split, :]
    X_train = X[split:, :, :, :]
    y_train = y[split:, :]

    return X_train, y_train, X_test, y_test

def real_to_cdf(y, sigma=1e-10):
    """
    Utility function for creating CDF from real number and sigma (uncertainty measure).
    :param y: array of real values
    :param sigma: uncertainty measure. The higher sigma, the more imprecise the prediction is, and vice versa.
    Default value for sigma is 1e-10 to produce step function if needed.
    """
    cdf = np.zeros((y.shape[0], 600))
    for i in range(y.shape[0]):
        cdf[i] = norm.cdf(np.linspace(0, 599, 600), y[i], sigma)
    return cdf

def crps(true, pred):
    """
    Calculation of CRPS.
    :param true: true values (labels)
    :param pred: predicted values
    """
    return np.sum(np.square(true - pred)) / true.size

def train(path_in,path_out,model_type):
    """
    Training systole and diastole models.
    """
    print('Loading and compiling models...')
    print model_type
    model_systole = get_model(model_type)
    model_diastole = get_model(model_type)

    print('Loading training data...')
    diastoleX, diastoleY, systoleX, systoleY = load_data(path_in)

    # split to training and test
    diastoleX_train, diastoleY_train, diastoleX_test, diastoleY_test = split_data(diastoleX, diastoleY, split_ratio=0.2)
    systoleX_train, systoleY_train, systoleX_test, systoleY_test = split_data(systoleX, systoleY, split_ratio=0.2)

    nb_iter = 200
    epochs_per_iter = 1
    batch_size = 32
    calc_crps = 1  # calculate CRPS every n-th iteration (set to 0 if CRPS estimation is not needed)

    # remember min val. losses (best iterations), used as sigmas for submission
    min_val_loss_systole = sys.float_info.max
    min_val_loss_diastole = sys.float_info.max
 
    loss_systole_list = []
    loss_diastole_list = []
    val_loss_systole_list = [] 
    val_loss_diastole_list = []

    print('-'*50)
    print('Training...')
    print('-'*50)

    for i in range(nb_iter):
        print('-'*50)
        print('Iteration {0}/{1}'.format(i + 1, nb_iter))
        print('-'*50)

        print('Fitting systole model...')
        hist_systole = model_systole.fit(systoleX_train, systoleY_train[:, 0], shuffle=True, nb_epoch=epochs_per_iter,
                                         batch_size=batch_size, validation_data=(systoleX_test, systoleY_test[:, 0]))

        print('Fitting diastole model...')
        hist_diastole = model_diastole.fit(diastoleX_train, diastoleY_train[:, 0], shuffle=True, nb_epoch=epochs_per_iter,
                                           batch_size=batch_size, validation_data=(diastoleX_test, diastoleY_test[:, 0]))
	
        # sigmas for predicted data, actually loss function values (RMSE)
        loss_systole = hist_systole.history['loss'][-1]
        loss_diastole = hist_diastole.history['loss'][-1]
        val_loss_systole = hist_systole.history['val_loss'][-1]
        val_loss_diastole = hist_diastole.history['val_loss'][-1]
	loss_systole_list.append(loss_systole)
	loss_diastole_list.append(loss_diastole)
	val_loss_systole_list.append(val_loss_systole)
        val_loss_diastole_list.append(val_loss_diastole)

        if calc_crps > 0 and i % calc_crps == 0:
            print('Evaluating CRPS...')
            pred_systole = model_systole.predict(systoleX_train, batch_size=batch_size, verbose=1)
            pred_diastole = model_diastole.predict(diastoleX_train, batch_size=batch_size, verbose=1)
            val_pred_systole = model_systole.predict(systoleX_test, batch_size=batch_size, verbose=1)
            val_pred_diastole = model_diastole.predict(diastoleX_test, batch_size=batch_size, verbose=1)

            # CDF for train and test data (actually a step function)
            cdf_train = real_to_cdf(np.concatenate((systoleY_train[:, 0], diastoleY_train[:, 0])))
            cdf_test = real_to_cdf(np.concatenate((systoleY_test[:, 0], diastoleY_test[:, 0])))

            # CDF for predicted data
            cdf_pred_systole = real_to_cdf(pred_systole, loss_systole)
            cdf_pred_diastole = real_to_cdf(pred_diastole, loss_diastole)
            cdf_val_pred_systole = real_to_cdf(val_pred_systole, val_loss_systole)
            cdf_val_pred_diastole = real_to_cdf(val_pred_diastole, val_loss_diastole)

            # evaluate CRPS on training data
            crps_train = crps(cdf_train, np.concatenate((cdf_pred_systole, cdf_pred_diastole)))
            print('CRPS(train) = {0}'.format(crps_train))

            # evaluate CRPS on test data
            crps_test = crps(cdf_test, np.concatenate((cdf_val_pred_systole, cdf_val_pred_diastole)))
            print('CRPS(test) = {0}'.format(crps_test))

        print('Saving weights...')
        # save weights so they can be loaded later
        model_systole.save_weights(path_out+model_type+'/'+'weights_systole.hdf5', overwrite=True)
        model_diastole.save_weights(path_out+model_type+'/'+'weights_diastole.hdf5', overwrite=True)

        # for best (lowest) val losses, save weights
        if val_loss_systole < min_val_loss_systole:
            min_val_loss_systole = val_loss_systole
            model_systole.save_weights(path_out+model_type+'/'+'weights_systole_best.hdf5', overwrite=True)

        if val_loss_diastole < min_val_loss_diastole:
            min_val_loss_diastole = val_loss_diastole
            model_diastole.save_weights(path_out+model_type+'/'+'weights_diastole_best.hdf5', overwrite=True)

        # save best (lowest) val losses in file (to be later used for generating submission)
        with open('val_loss.txt', mode='w+') as f:
            f.write(str(min_val_loss_systole))
            f.write('\n')
            f.write(str(min_val_loss_diastole))
    
    np.save(path_out+model_type+'/'+'loss_systole_list.npy',np.array(loss_systole_list))
    np.save(path_out+model_type+'/'+'loss_diastole_list.npy',np.array(loss_diastole_list))
    np.save(path_out+model_type+'/'+'val_loss_systole_list.npy',np.array(val_loss_systole_list))
    np.save(path_out+model_type+'/'+'val_loss_diastole_list.npy',np.array(val_loss_diastole_list))
    
    model_systole_best = get_model(model_type)
    model_systole_best.load_weights(path_out+model_type+'/'+'weights_systole_best.hdf5')
    systole_pred = model_systole_best.predict(systoleX, batch_size=batch_size, verbose=1)
    np.save(path_out+model_type+'/'+'systole_pred.npy',systole_pred)

    model_diastole_best = get_model(model_type)
    model_diastole_best.load_weights(path_out+model_type+'/'+'weights_diastole_best.hdf5')
    diastole_pred = model_diastole_best.predict(diastoleX, batch_size=batch_size, verbose=1)
    np.save(path_out+model_type+'/'+'diastole_pred.npy',diastole_pred)

def test(path_in,path_out,model_type):
	print model_type
 	print 'Test on validation data'	
	print 'Loading Data ...'
	diastoleX, diastoleY, systoleX, systoleY = load_data(path_in)
	print 'Data Loaded'
	batch_size = 32

	print 'Predict Systole'
	model_systole_val = get_model(model_type)
	model_systole_val.load_weights(path_out+model_type+'/'+'weights_systole_best.hdf5')
	systole_pred = model_systole_val.predict(systoleX, batch_size=batch_size, verbose=1)
	np.save(path_out+model_type+'/'+'systole_val_pred.npy',systole_pred)
	np.save(path_out+model_type+'/'+'systole_val_real.npy',systoleY)
	
	print 'Predict Diastole'
        model_diastole_val = get_model(model_type)
        model_diastole_val.load_weights(path_out+model_type+'/'+'weights_diastole_best.hdf5')
        diastole_pred = model_diastole_val.predict(diastoleX, batch_size=batch_size, verbose=1)
        np.save(path_out+model_type+'/'+'diastole_val_pred.npy',diastole_pred)
	np.save(path_out+model_type+'/'+'diastole_val_real.npy',diastoleY)

if __name__ == "__main__":
	path_in = sys.argv[1]
	path_out = sys.argv[2]
	mode = sys.argv[3]
	
	model_list = ['model1','model2','model3']
	time_list = []	
	time_iter = 0
	
	if mode == 'train':
		for model_type in model_list:
			start = time.time()
			train(path_in,path_out,model_type)
			time_list.append(time.time() - start)
                	print ('Time elapsed')
               		print time_list[time_iter]
                	time_iter += 1

	if mode == 'test':
		for model_type in model_list:
			start = time.time()
			test(path_in,path_out,model_type)
			time_list.append(time.time() - start)
			print ('Time elapsed')
			print time_list[time_iter]
			time_iter += 1
