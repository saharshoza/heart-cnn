# Usage: python extract_features.py <caffe_path> <root_path_to_labelled_data> <path_to_pretrained_weights> <path_to_model> <number_of_examples>
# Input: caffe_path: path to the caffe root folder
#	 root_path: path to which the output of data_input.py was stored 
#	 path_to_pretrained_weights: path will be typically be models/<model_dir>/<Huge file you downloaded>
#	 path_to_model: path to the deploy.prototxt Change the number of examples in the deploy.prototxt to 50 for this code to work
#	 number_of_examples: 500 in the case of train and 200 for validation in this dataset
# Output: Save the extracted features as Systole.h5 and Diastole.h5 in root_path_to_labelled_data/extracted_feat

import numpy as np
import cv2
import sys
import os
from skimage.restoration import denoise_tv_chambolle
from scipy import ndimage
import tables

ANGLE_RANGE = 15
HEIGHT_SHIFT = 10
WIDTH_SHIFT = 10
CHUNK_SIZE = 50

# Get mean image of the dataset (sax1_max, sax2_max, ...)
# Shift and rotate the images here as well to make the model resistant to over fitting
# Also return image dataset. Misleading function name?
def get_mean_image(fileout,filein):
        f = open(filein)
        rows = f.read()
        f.close()
        image_row = rows.split('\n')
	#print image_row
	image_row = [idx for idx in image_row if idx not in '']
	image_row = image_row[1:]
        image_path = [img_loc.split(' ')[0] for img_loc in image_row]
	image_dataset = np.zeros((len(image_row),224,224,3))
        mean = np.zeros(cv2.imread(image_path[0]).shape)
        mean_list = []
        mean = np.zeros((224,224,3))
        for img in range(0,len(image_path)-1):
                data_X = cv2.resize(cv2.imread(image_path[img]), (224,224))
		data_X = denoise_tv_chambolle(data_X, weight=0.1, multichannel=False)
		angle = np.random.randint(-ANGLE_RANGE, ANGLE_RANGE)
		X_rotated = ndimage.rotate(data_X, angle, reshape=False, order=1)
		h_shift = np.random.randint(-HEIGHT_SHIFT,HEIGHT_SHIFT)
		w_shift = np.random.randint(-WIDTH_SHIFT,WIDTH_SHIFT)
		image_dataset[img] = ndimage.shift(X_rotated, (0, h_shift, w_shift), order=0)
		mean += image_dataset[img]
        mean = mean/img
	print caffe.proto.caffe_pb2.BlobProto
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.shape.dim.extend(map(int, mean.shape))
        blob.data.extend(mean.flatten().tolist())
        blob.channels = mean.shape[2]
        blob.height = mean.shape[0]
        blob.width = mean.shape[1]
        blob.num = 1
        arr = np.array( caffe.io.blobproto_to_array(blob) )
        out = arr[0]
        np.save( fileout , out )
	print image_dataset.shape
	return image_dataset

# Create transformer object for each dataset for its mean
def get_transformer(in_file,net):
	mu = np.load(in_file)
	mu = mu.mean(1).mean(1)
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
	transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
	transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
	transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
	return transformer

# Create net for the model and weights supplied
def create_net(caffe_root,weight_path,model_path):
	if os.path.isfile(caffe_root + weight_path):
    		print 'CaffeNet found.'
	else:
    		print 'Get net. This will fail'
	caffe.set_mode_cpu()
	model_def = caffe_root + model_path
	model_weights = caffe_root + weight_path
	net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
	return net

if __name__ == "__main__":
	caffe_root = sys.argv[1]
	root_path = sys.argv[2]
	weight_path = sys.argv[3]
	model_path = sys.argv[4]
	num_examples = int(sys.argv[5])
	
	## Add caffe path to sys path. Then import caffe module
	sys.path.insert(0, caffe_root + 'python')
	caffe = map(__import__, ['caffe'])[0]
	caffe_proto = map(__import__, ['caffe.proto.caffe_pb2'])[0]
	input_path = root_path + 'labelled_data/'
	subpath_list = ['mean_image/','extracted_feat/']
	file_list = ['sax1_max.csv','sax2_max.csv','sax3_max.csv','ch_max.csv','sax1_min.csv','sax2_min.csv','sax3_min.csv','ch_min.csv']

	## If you have a GPU, knock yourself out
	caffe.set_mode_cpu()
	net = create_net(caffe_root,weight_path,model_path)
	if not os.path.exists(root_path+subpath_list[0]):
                os.makedirs(root_path+subpath_list[0])
	if not os.path.exists(root_path+subpath_list[1]):
                os.makedirs(root_path+subpath_list[1])
	diastole_list = ['sax1_max.csv','sax2_max.csv','sax3_max.csv','ch_max.csv']
	systole_list = ['sax1_min.csv','sax2_min.csv','sax3_min.csv','ch_min.csv']
	file_list = [diastole_list, systole_list]

	for list_name in file_list:
		if '_max' in list_name[0]:
			final_file = 'diastoleX.h5'
		else:
			final_file = 'systoleX.h5'
		final_file = root_path+subpath_list[1]+final_file
		print list_name
		feat_list = []
		for idx in list_name:

			print idx
			fileout = root_path+subpath_list[0]+idx
			fileout = fileout.replace('.csv','.npy')

			# Get mean file and image dataset
			image_dataset = get_mean_image(fileout,input_path+idx)
			print 'mean file created'
			transformer = get_transformer(fileout,net)
			transformed_image_dataset = np.zeros(image_dataset.transpose(0,3,2,1).shape)
			feat = np.zeros((num_examples,512,14,14))
			print 'feat shape'
			print feat.shape

			# Pre process the images using transformer object of net
			for num_examples_iter in range(0,image_dataset.shape[0]):
				transformed_image_dataset[num_examples_iter] = transformer.preprocess('data', image_dataset[num_examples_iter])
			print 'Image transformed'
			image_dataset = None
			for chunk_iter in range(0,transformed_image_dataset.shape[0]/CHUNK_SIZE):
				print chunk_iter
				net.blobs['data'].data[...] = transformed_image_dataset[chunk_iter*(CHUNK_SIZE):(chunk_iter + 1)*CHUNK_SIZE]
				output = net.forward()
				print 'Forward pass done'
				feat[chunk_iter*(CHUNK_SIZE):(chunk_iter + 1)*CHUNK_SIZE] = net.blobs['conv5_4'].data
			feat_list.append(feat)

		# Concatenate features from all 4 buckets
		_temp1 = np.concatenate((feat_list[0],feat_list[1]),axis=2)
		print _temp1.shape
		_temp2 = np.concatenate((feat_list[2],feat_list[3]),axis=2)
		print _temp2.shape
		all_feat = np.concatenate((_temp1,_temp2),axis=3)
		print all_feat.shape

		# Write to hdf5 files
		f = tables.open_file(final_file, mode='w')
		atom = tables.Float64Atom()
		array_c = f.create_earray(f.root, 'data', atom, (0, 512,28,28))
    		array_c.append(all_feat)
		f.close()
