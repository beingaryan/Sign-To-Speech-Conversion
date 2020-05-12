from keras.preprocessing import image
import pyautogui
from playsound import playsound
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import regularizers
#print(os.listdir("../input"))
train_dir = 'D:\GESTURE\dataset'



def load_unique():
    
    
    size_img = 64,64
    images_for_plot = []
    labels_for_plot = []
    count1=0
    count2=0
    for folder in os.listdir(train_dir):
        count1=count1+1
        for file in os.listdir(train_dir + '/' + folder):
            count2=count2+1
            filepath = train_dir +'/'+folder+'/'+ file
            print(filepath)
            image = cv2.imread(filepath)
            final_img = cv2.resize(image, size_img)
            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            images_for_plot.append(final_img)
            labels_for_plot.append(folder)
            break
    print("count1" + str(count1))
    print("count2" + str(count2))
     
    return images_for_plot, labels_for_plot

images_for_plot, labels_for_plot = load_unique()
print("unique_labels = ", labels_for_plot)


# In[5]:

labels_dict = {'1':0,'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'9':8,'10':9,'11':10,'12':11,'13':12,'14':13,'15':14,'16':15,
'17':16,'18':17,'19':18,'20':19,'21':20,'22':21,'23':22,'24':23,'25':24,'26':25}


def load_data():
    images = []
    labels = []
    size = 64,64
    print("LOADING DATA FROM : ",end = "")
    for folder in os.listdir(train_dir):
        print(folder, end = ' | ')
        for image in os.listdir(train_dir + "/" + folder):
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
            temp_img = cv2.resize(temp_img, size)
            #images.append(temp_img)
            if folder == 'A':
                
                images.append(temp_img)
                labels.append(labels_dict['1'])
            elif folder == 'B':
                images.append(temp_img)
                labels.append(labels_dict['2'])
            elif folder == 'C':
                images.append(temp_img)
                labels.append(labels_dict['3'])
            elif folder == 'D':
                images.append(temp_img)
                labels.append(labels_dict['4'])
            elif folder == 'E':
                images.append(temp_img)
                labels.append(labels_dict['5'])
            elif folder == 'F':
                images.append(temp_img)
                labels.append(labels_dict['6'])
            elif folder == 'G':
                images.append(temp_img)
                labels.append(labels_dict['7'])
            elif folder == 'H':
                images.append(temp_img)
                labels.append(labels_dict['8'])
            elif folder == 'I':
                images.append(temp_img)
                labels.append(labels_dict['9'])
            elif folder == 'J':
                images.append(temp_img)
                labels.append(labels_dict['10'])
             
            elif folder == 'K':
                images.append(temp_img)
                labels.append(labels_dict['11'])
            elif folder == 'L':
                images.append(temp_img)
                labels.append(labels_dict['12'])
            elif folder == 'M':
                images.append(temp_img)
                labels.append(labels_dict['13'])
            elif folder == 'N':
                images.append(temp_img)
                labels.append(labels_dict['14'])
            elif folder == 'O':
                images.append(temp_img)
                labels.append(labels_dict['15'])
            elif folder == 'P':
                images.append(temp_img)
                labels.append(labels_dict['16'])
            elif folder == 'Q':
                images.append(temp_img)
                labels.append(labels_dict['17'])
            elif folder == 'R':
                images.append(temp_img)
                labels.append(labels_dict['18'])
            elif folder == 'S':
                images.append(temp_img)
                labels.append(labels_dict['19'])
            elif folder == 'T':
                images.append(temp_img)
                labels.append(labels_dict['20'])
            elif folder == 'U':
                images.append(temp_img)
                labels.append(labels_dict['21'])
            elif folder == 'V':
                images.append(temp_img)
                labels.append(labels_dict['22'])
            elif folder == 'W':
                images.append(temp_img)
                labels.append(labels_dict['23'])
            elif folder == 'X':
                images.append(temp_img)
                labels.append(labels_dict['24'])
            elif folder == 'Y':
                images.append(temp_img)
                labels.append(labels_dict['25'])
            elif folder == 'Z':
                images.append(temp_img)
                labels.append(labels_dict['26'])
            


    images = np.array(images)
    print("length")
    print(len(images))
    print(len(labels))
    #print(images)
    #print(labels)
    images = images.astype('float32')/255.0
    
    labels = keras.utils.to_categorical(labels)   #one-hot encoding
    
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = 0.1)
    
    print()
    print('Loaded', len(X_train),'images for training,','Train data shape =',X_train.shape)
    print('Loaded', len(X_test),'images for testing','Test data shape =',X_test.shape)
    
    return X_train, X_test, Y_train, Y_test


# In[6]:

print("***********************")
X_train, X_test, Y_train, Y_test = load_data()
#print(X_train)
print("*************************")
#print(Y_train)
print("*************************")
#print(X_test)
print("*************************")

#print(Y_test)


# In[41]:


def build_model():
    print(" in build model ")
    
    model = Sequential()
    
    model.add(Conv2D(64, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (64,64,3)))
    model.add(Conv2D(32, kernel_size = 3, padding = 'same', strides = 2, activation = 'relu'))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(64, kernel_size = 3, padding = 'same', strides = 2, activation = 'relu'))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(128, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(256, kernel_size = 3, padding = 'same', strides = 2 , activation = 'relu'))
    model.add(MaxPool2D(3))
    
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(26, activation = 'softmax'))
    
    model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"])
    
    print("MODEL CREATED")
    #model.summary()
    
    return model


# In[42]:


def fit_model():
    print(" in fitmodel")
    history = model.fit(X_train, Y_train, batch_size = 32, epochs = 50, validation_split = 0.1)
    return history


# In[43]:

print("going to build")
model = build_model()
print("after build moel")


# In[44]:

print("going to history")
model_history = fit_model()
print("after history")

if model_history:
  print('Final Accuracy: {:.2f}%'.format(model_history.history['accuracy'][4] * 100))
  print('Validation Set Accuracy: {:.2f}%'.format(model_history.history['val_accuracy'][4] * 100))


# In[46]:


model.save("trained_model.h5")


# In[48]:


#model.save("trained_model.model")


# In[39]:


#evaulate_metrics=model.e(X_test,Y_test)


# In[49]:


#evaluate_metrics=model.evaluate(X_test,Y_test)
#print(evaluate_metrics)

# In[ ]:




