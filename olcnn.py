#!/usr/bin/env python
# coding: utf-8

# In[43]:


# Import libraries
import os,cv2
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K 
from keras.optimizers import Adam


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from keras.optimizers import SGD,RMSprop,adam
K.set_image_data_format('channels_last')
#%%
#SOMEWHAT RELATED TO PATH
PATH = 'D:/New Project/OLCNN/'
# Define data path
data_path = PATH + '/RRNN/0'
data_dir_list = os.listdir(data_path)
img_rows=224
img_cols=224
num_channel=1
num_epoch=160

# Define the number of classes
num_classes = 5

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		input_img_resize=cv2.resize(input_img,(224,224))
		img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)

if num_channel==1:
	if K.image_data_format=='channels_last':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=3) 
		print (img_data.shape)
		
else:
	if K.image_data_format=='channels_last':
		img_data=np.rollaxis(img_data,3,1)
		print (img_data.shape)
		


# In[40]:


# Assigning Labels

# Define the number of classes
num_classes = 5

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:286]=0
labels[287:489]=1
labels[490:1153]=2
labels[1154:1644]=3
labels[1645:]=4
	  
names = ['pull','push','sitDown','standUp','throw']
	  
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


# In[41]:


# Defining the model
input_shape=img_data[0].shape
					
model = Sequential()
model.add(Conv2D(8,3,padding="same", input_shape = (224,224,1))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Conv2D(12,3,padding="same")) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Conv2D(24,3,padding="same")) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))  
opt = Adam(lr=0.001,decay=0.000005/num_epoch)
#build your own cnn
""" model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1)) 
model.add(Activation('sigmoid'))  
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])"""
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=["accuracy"])

# Viewing model_configuration

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable


# In[42]:


# Training
hist = model.fit(X_train, y_train, batch_size=20, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))

#hist = model.fit(X_train, y_train, batch_size=32, nb_epoch=20,verbose=1, validation_split=0.2)'
# Training with callbacks

#visualize 2
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs_range = range(160)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[ ]:




