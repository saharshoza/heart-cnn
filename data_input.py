# Usage: python data_input.py <path_to_dataset> <path_to_label> <path_to_output> <mode>

# Inputs:
# path_to_dataset: Path to input dataset is stored in the same format as provided on kaggle. ie 1/study/<views>/<images>
# path_to_label: Path to input labels in csv format
# path_to_output: Path to which the output csv files must be stored. 
# mode: The script can be run as individual modules:
#	step1: Identifies the systole and diastole image for each patient and stores in path_to_output/max_min_select/
#	step2: Selects 1 image at random from each bucket and stores in path_to_output/random_select/
#	step3: Attaches a label to each of the images selected above and stores in path_to_output/labelled_data/
#	all: Perform all 3 steps at once

# Outputs:
# The output of each intermediate step is stored in a separate folder. 
# Each folder contains 8 csv files.
# sax1_max.csv, sax2_max.csv, sax3_max.csv, ch_max.csv for diastole
# sax1_min.csv, sax2_min.csv, sax3_min.csv, ch_min.csv for systole
# The final set of 8 is stored in path_to_output/labelled_data

# Import all dependencies
# Install ImageMagick
import pandas
import cv2
import time
import subprocess
import os
import csv
import sys
import numpy as np
import pandas
import random

# Create headers for each csv file
def create_header(path_to_output,output_list,num_col_list,subpath):
	if not os.path.exists(path_to_output+subpath):
		os.makedirs(path_to_output+subpath)
	for i in range(0,len(output_list)):
		with open(path_to_output+subpath+output_list[i], 'wb') as csvfile:
			train_writer = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
			header = ['num']
			header_col = [str(col_i) for col_i in range(1,num_col_list[i]+1)]
			header.extend(header_col)
    			train_writer.writerow(header)

# Create a list of each of the 4 buckets.
# For sax, arrange all sax views in a folder in ascending order and split it into 3 parts. 
# The 3 sax buckets are returned as a sax_chunk
# Returns: sax_chunk containing 3 sax_buckets and ch_list containing the ch_bucket
def get_view_lists(img_num,path_to_dataset):
	path_to_example = path_to_dataset+img_num+'/study/'
    	os.chdir(path_to_example)
	process = subprocess.Popen("ls",shell=True, stdout=subprocess.PIPE,)
    	stdout = process.communicate()[0].split('\n')
    	view_list = [idx for idx in stdout if idx not in '']
    	sax_num = [int(idx.split('sax_')[1]) for idx in view_list if 'sax_' in  idx]
    	sax_num.sort()
    	sax_list = ['sax_'+str(idx) for idx in sax_num]
    	if int(round(float(len(sax_list))/3)) == 0:
        	sax_chunks = [sax_list, sax_list, sax_list]
    	else:
        	sax_chunks = [sax_list[i:i+int(round(float(len(sax_list))/3))] for i in range(0, len(sax_list), int(round(float(len(sax_list))/3)))]
    	if len(sax_chunks) > 3:
        	sax_chunks[2].extend(sax_chunks[3])
        	del sax_chunks[3]
    	ch_list = [idx for idx in view_list if 'ch_' in idx]
	return sax_chunks, ch_list

# Perform otsu thresholding. Compute brightness and write the brightest and darkest to _max.csv and _min.csv respectively
def max_min_write_csv(img_num,path_to_example,view,axis,path_to_output):
	min_list = [img_num]
        max_list = [img_num]
	for view_idx in view:
        	path_to_view = path_to_example+view_idx+'/'
        	os.chdir(path_to_view)
        	mogrify = subprocess.Popen("mogrify -format jpeg *.dcm",shell=True, stdout=subprocess.PIPE,)
        	time.sleep(1)
        	grep = subprocess.Popen("ls | grep jpeg", shell=True, stdin=mogrify.stdout, stdout=subprocess.PIPE,)
        	mogrify.stdout.close()
        	stdout = grep.communicate()[0].split('\n')
        	mogrify.wait()
        	grep.stdout.close()
        	img_list = [idx for idx in stdout if idx not in '' and '~' not in idx]
       	 	img_abs_path = [os.getcwd()+'/'+idx for idx in img_list]
       	 	bright_vol = np.zeros(len(img_abs_path))
        	for img_num in range(0,len(img_abs_path)):
            		img = cv2.imread(img_abs_path[img_num])
            		_, otsu_img = cv2.threshold(cv2.cvtColor( img, cv2.COLOR_BGR2GRAY ),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            		hist = np.bincount(otsu_img.reshape(-1))
            		bright_vol[img_num] = (float(hist[255])/float(hist[255] + hist[0]))*100
        	max_idx = np.argmax(bright_vol)
        	min_idx = np.argmin(bright_vol)
        	min_list.append(img_abs_path[min_idx])
        	max_list.append(img_abs_path[max_idx])
	with open(path_to_output+axis+'_max.csv', 'a') as csvfile:
		train_writer = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
        	train_writer.writerow(max_list)
        with open(path_to_output+axis+'_min.csv', 'a') as csvfile:
                train_writer = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
                train_writer.writerow(min_list)

# Identify systole and diastole images
def max_min_selection(path_to_dataset,path_to_output,out_list,subpath):
	num_col_list = [8,8,8,3,8,8,8,3]
	os.chdir(path_to_dataset)
	process = subprocess.Popen("ls",shell=True, stdout=subprocess.PIPE,)
	stdout = process.communicate()[0].split('\n')
	example_list = [i for i in stdout if i not in '']
	create_header(path_to_output,out_list,num_col_list,subpath)
	for img_num in example_list:
		sax_chunks,ch_list = get_view_lists(img_num,path_to_dataset)
		max_min_write_csv(img_num,path_to_dataset+img_num+'/study/',ch_list,'ch',path_to_output+subpath)
		sax_vol = 0
		for sax_list in sax_chunks:
			sax_vol += 1
			max_min_write_csv(img_num,path_to_dataset+img_num+'/study/',sax_list,'sax'+str(sax_vol),path_to_output+subpath)

# Select 1 image at random from each bucket
# The random index chosen may sometimes point to NaN (as the number of views per patient is variable)
# If this happens for less than 2% of the examples, pick the first view of that bucket.
def random_view_selection(path_to_output,subpath_list,in_list):
        if not os.path.exists(path_to_output+subpath_list[1]):
                os.makedirs(path_to_output+subpath_list[1])
	for in_file in in_list:
		df = pandas.read_csv(path_to_output+subpath_list[0]+in_file).set_index('num')
		nan_percentage = 100
		while nan_percentage > 2:
			df_rand = df[str(random.randint(1,int(df.columns.values[-1])))].sort_index(axis=0)
			nan_idx = [key for (key, value) in pandas.isnull(df_rand).to_dict().items() if value == True]
                	nan_percentage = (float(len(nan_idx))/500)*100
        	for idx in nan_idx:
                	df_rand.loc[idx] = df.loc[idx,'1']
		df_new = df_rand.to_frame(name='data')
		df_new.to_csv(path_to_output+subpath_list[1]+in_file)

# Attach a label from the csv file to each of randomly selected images per bucket
def attach_label(path_to_label,subpath_list,in_list):
        if not os.path.exists(path_to_output+subpath_list[2]):
                os.makedirs(path_to_output+subpath_list[2])
	label = pandas.read_csv(path_to_label).set_index('Id')
	for in_file in in_list:
		df = pandas.read_csv(path_to_output+subpath_list[1]+in_file)
		df = df.set_index('num')
		if '_max' in in_file:
			df_join = label.join(df)[['data','Diastole']]
			null_list = df_join[pandas.isnull(df_join).any(axis=1)].index.values.tolist()
			corr_list = [i-1 for i in null_list]
			for i in range(0,len(corr_list)):
				df_join.loc[null_list[i],'data'] = df_join.loc[corr_list[i],'data']
			df_join = df_join[pandas.notnull(df_join['data'])]
			df_join.to_csv(path_to_output+subpath_list[2]+in_file,sep=' ',index=False)
		else:
			df_join = label.join(df)[['data','Systole']]
                        null_list = df_join[pandas.isnull(df_join).any(axis=1)].index.values.tolist()
                        corr_list = [i-1 for i in null_list]
                        for i in range(0,len(corr_list)):
                                df_join.loc[null_list[i],'data'] = df_join.loc[corr_list[i],'data']
                        df_join = df_join[pandas.notnull(df_join['data'])]
                        df_join.to_csv(path_to_output+subpath_list[2]+in_file,sep=' ',index=False)
	
if __name__ == "__main__":
	path_to_dataset = sys.argv[1]
	path_to_label = sys.argv[2]
	path_to_output = sys.argv[3]
	mode = sys.argv[4]
	# This is pre-defined
	out_list = ['sax1_max.csv','sax2_max.csv','sax3_max.csv','ch_max.csv','sax1_min.csv','sax2_min.csv','sax3_min.csv','ch_min.csv']
	subpath_list = ['max_min_select/','random_select/','labelled_data/']
	# Identify systole and diastole images in every view folder
	if mode == 'step1' or mode == 'all':
		max_min_selection(path_to_dataset,path_to_output,out_list,subpath_list[0])
	# Pick an image at random from every view bucket
	if mode == 'step2' or mode == 'all':
		random_view_selection(path_to_output,subpath_list,out_list)
	# Attach the label from input csv to each of the randomly picked images
	if mode == 'step3' or mode == 'all':
		attach_label(path_to_label,subpath_list,out_list)
