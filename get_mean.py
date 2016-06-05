## Usage: python get_mean.py <input_csv1> <output_npy1> <input_csv2> <output_npy2> ...
## Inputs: input_csv is the path to csv stored in labelled_data/ after data_input.py
##	   path to output_npy where mean of the dataset must be stored 

## Import all dependencies
import numpy as np
import cv2
import sys

## Enter the path to your caffe installation here
caffe_root = '/home/ubuntu/deep_learning/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import caffe.proto.caffe_pb2 as caffe_proto


def get_mean_image(fileout,filein):
	# Read the csv and pick only the image path from the csv
	f = open(filein)
	rows = f.read()
	f.close()
	image_row = rows.split('\n')
	image_path = [img_loc.split(' ')[0] for img_loc in image_row]
	mean = np.zeros(cv2.imread(image_path[0]).shape)
	mean_list = []
	mean = np.zeros((256,256,3))
	for img in range(0,len(image_path)-1):
		mean += cv2.resize(cv2.imread(image_path[img]), (256,256))
	mean = mean/img
	blob = caffe_proto.BlobProto()
    	blob.shape.dim.extend(map(int, mean.shape))
    	blob.data.extend(mean.flatten().tolist())
  	# legacy
    	blob.channels = mean.shape[2]
    	blob.height = mean.shape[0]
    	blob.width = mean.shape[1]
    	blob.num = 1
	arr = np.array( caffe.io.blobproto_to_array(blob) )
	print arr.shape
	out = arr[0]
	np.save( fileout , out )	

if __name__ == "__main__":
	for i in range(1,len(sys.argv),2):
		get_mean_image(sys.argv[i],sys.argv[i+1])
	
