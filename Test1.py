import numpy as np
from skimage import feature
import os, cv2, csv, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='what folder to use for HOG description',
choices=['RR'])
args = vars(parser.parse_args())
images = []
labels = []

def preProcessing():
    # get all the image folder paths
    image_paths = os.listdir(f"{args['path']}")
    
    for path in image_paths:
        # get all the image names
        all_images = os.listdir(f"{args['path']}/{path}")
        
        # iterate over the image names, get the label
        for image in all_images:
            image_path = f"{args['path']}/{path}/{image}"
            image = cv2.imread(image_path)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (80, 120))

            # get the HOG descriptor for the image
            hog_desc = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

            #print(np.size(hog_desc))
            # update the data and labels
            images.append(hog_desc)
            labels.append(path)

    # writing the data into the file 
    np.savetxt("labels.csv", labels, delimiter ="", fmt ='% s')  
    file = open('images.csv', 'w+', newline ='') 
    with file:     
        write = csv.writer(file)
        write.writerows(images) 
    file.close()

preProcessing()