# heart-cnn
Predict heart volume using a CNN model trained over MRI images of 500 patients

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

 
