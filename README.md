# heart-cnn
Predict heart volume using a CNN model trained over MRI images of 500 patients. A detailed explanation of all the steps involved can be found at http://saharshoza.github.io/heart-cnn/

# ViewImages.ipynb
Change the directory in the notebook to a single view of a patient in  your training data directory. The notebook will print the images and the systolic and diastolic images in the set of 30 images in that view.

# data\_input.py
`python data_input.py path_to_dataset path_to_label  path_to_output mode`  

Inputs:  
path\_to\_dataset: Path to input dataset is stored in the same format as provided on kaggle. ie 1/study/views/images  
path\_to\_label: Path to input labels in csv format  
path\_to\_output: Path to which the output csv files must be stored  
mode: The script can be run as individual modules:  
	step1: Identifies the systole and diastole image for each patient and stores in path_to_output/max_min_select/  
	step2: Selects 1 image at random from each bucket and stores in path_to_output/random_select/  
	step3: Attaches a label to each of the images selected above and stores in path_to_output/labelled_data/  
	all: Perform all 3 steps at once  
  
Outputs:  
The output of each intermediate step is stored in a separate folder.   
Each folder contains 8 csv files.  
sax1\_max.csv, sax2\_max.csv, sax3\_max.csv, ch\_max.csv for diastole  
sax1\_min.csv, sax2\_min.csv, sax3\_min.csv, ch\_min.csv for systole  
The final set of 8 is stored in path\_to\_output/labelled\_data  

# get\_mean.py
`python get_mean.py <input_csv1> <output_npy1> <input_csv2> <output_npy2> ...`  

Inputs: input\_csv is the path to csv stored in labelled\_data/ after data\_input.py
        path to output_npy where mean of the dataset must be stored 

# ViewExtractedFeatures.ipynb
Use the notebook to visualize the features at any layer of the VGG\_ILSVRC\_19\_layers pre trained net 

# extract\_features.py
`python extract_features.py <caffe_path> <root_path_to_labelled_data> <path_to_pretrained_weights> <path_to_model> <number_of_examples>`  
Input: caffe\_path: path to the caffe root folder  
       root_path: path to which the output of data_input.py was stored  
       path_to_pretrained_weights: path will be typically be models/<model_dir>/<Huge file you downloaded>  
       path_to_model: path to the deploy.prototxt Change the number of examples in the deploy.prototxt to 50 for this code to work  
       number_of_examples: 500 in the case of train and 200 for validation in this dataset  
Output: Save the extracted features as Systole.h5 and Diastole.h5 in root\_path\_to\_labelled\_data/extracted\_feat  

# get\_labels.py
`python get_labels.py <input_path_to_labelled_data> <output_path_to_labels>`  
Use this to store the labels of the images in hdf5 format

# keras\_model.py
`python keras_model.py <path_to_input> <path_to_output> <mode>`  
Input: path\_to\_input: path to the directory where extracted\_feat and labels are stored  
       path_to_output: path where train, val loss, test prediction for each model will be stored  
       mode: test or train  
Output: Store arrays for the train and validation loss at each iteration along with the actual versus predicted for validation images  
