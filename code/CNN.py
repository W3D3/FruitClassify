#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Imports
import numpy as np 
import cv2
import glob
import os
import matplotlib.pyplot as plt
#from mlxtend.plotting import plot_decision_regions
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

print(os.listdir("../input"))
dim = 100 # pixel dimension


# In[3]:


"""
fruits: array of selected fruits
data_type: "Training" or "Test"
label: overrides every label in the set
"""
def getYourFruits(fruits, data_type, label=None):
    images = []
    labels = []
    path = "../input/" + data_type + "/"
    
    for lbl, f in enumerate(fruits):
        p = path + f
        for image_path in glob.glob(os.path.join(p, "*.jpg")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (dim, dim))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            images.append(image)
            if label == None: 
                labels.append(lbl)
            else:
                labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


# In[4]:


fruits = ['Banana', 'Orange' , 'Kiwi']

#Get Images and Labels
x_train, y_train =  getYourFruits(fruits, 'Training')
x_test, y_test = getYourFruits(fruits, 'Test')


# In[5]:


#Let's visualize the first 10 training images!
import matplotlib.pyplot as plt

fig = plt.figure(figsize =(30,5))
for i in range(10):
    ax = fig.add_subplot(2,5,i+1,xticks=[],yticks=[])
    ax.imshow(np.squeeze(x_train[i]))


# In[6]:


# Let's confirm the number of classes
no_of_classes = len(np.unique(y_train))
no_of_classes


# In[7]:


print(y_train[0])
# target labels are numbers corresponding to class label. We need to change them to a vector of the elements.


# In[8]:


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train,no_of_classes)
y_test = np_utils.to_categorical(y_test,no_of_classes)
y_train[0] # Note that only one element has value 1(corresponding to its label) and others are 0.


# In[9]:


y_train[-1]


# In[ ]:


print('Training set shape : ',x_train.shape)

print('Test set shape : ',x_test.shape)

print('1st training image shape ',x_train[0].shape)


# In[ ]:


print('1st training image as array',x_train[0]) # don't worry if you see only 255s..


# In[ ]:


#Simple CNN from scratch - we are using 3 Conv layers followed by maxpooling layers.
# At the end we add dropout, flatten and some fully connected layers(Dense).

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K

model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = (3,3),input_shape=(100,100,3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 32,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 64,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 128,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(no_of_classes,activation = 'softmax'))
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print('Compiled!')


# In[ ]:


model.fit(x_train,y_train)


# In[ ]:


# evaluate and print test accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])


# In[ ]:




