#Usage: python get_labels.py <input_path_to_labelled_data> <output_path_to_labels>
import sys
import tables
import numpy as np

if __name__ == "__main__":
	path_in = sys.argv[1]
	path_out = sys.argv[2]
	filein_list = ['sax1_max.csv','sax1_min.csv']
	fileout_list = ['diastoleY.h5','systoleY.h5']
	for i in range(0,len(filein_list)): 
		
		filein = path_in+filein_list[i]
		fileout = path_out+fileout_list[i]
    
		f = open(filein)
        	rows = f.read()
        	f.close()
        	image_row = rows.split('\n')
        	image_row = [idx for idx in image_row if idx not in '']
        	image_row = image_row[1:]
        	image_label = [img_loc.split(' ')[1] for img_loc in image_row]
		image_label = [float(img_label) for img_label in image_label]
		img_array = np.array(image_label)
		img_array = img_array.reshape(1,len(image_label))
		print img_array.shape

        	f = tables.open_file(fileout, mode='w')
                atom = tables.Float64Atom()
                array_c = f.create_earray(f.root, 'data', atom, (0,len(image_label)))
                array_c.append(img_array)
                f.close()

	
