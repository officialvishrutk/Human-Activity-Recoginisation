import os
import cv2
import argparse

from sklearn.ensemble import RandomForestClassifier  #imports the random forest algorithm
from sklearn.datasets import make_classification           #imports random classifier generator
from skimage import feature

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='what folder to use for HOG description',
choices=['RGB'])
args = vars(parser.parse_args())
images = []
labels = []
# get all the image folder paths
image_paths = os.listdir(f"{args['path']}")
for path in image_paths:
    # get all the image names
    all_images = os.listdir(f"{args['path']}/{path}")
    # iterate over the image names, get the label
    for image in all_images:
        image_path = f"{args['path']}/{path}/{image}"
        image = cv2.imread(image_path)
        image = cv2.resize(image, (80, 120))
        # get the HOG descriptor for the image
        hog_desc = feature.hog(image, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

        # update the data and labels    
        images.append(hog_desc)
        labels.append(path)
model = RandomForestClassifier(n_estimators=46, random_state=0)
model.fit(images, labels)

print(model.predict)