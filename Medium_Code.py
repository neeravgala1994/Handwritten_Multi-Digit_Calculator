# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 09:52:34 2020

@author: 20044
"""

#--------------------------------------------------------------------------------
###CODE
#--------------------------------------------------------------------------------

#~~~1. Data set ~~~
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import seaborn as sns

np.random.seed(2)

'load the dataset'
dataset = pd.read_csv(r"C:\Users\20044\Desktop\Digital Recognizer\Dataset\digit + operators_2.csv")

"creating label"
y = dataset["label"]

"dropping label"
X = dataset.drop(labels = ["label"], axis = 1)

"deleting dataset to reduce memory usage"
del dataset

'overview of dataset'
g = sns.countplot(y)
y.value_counts()

'Grayscale normalization to reduce the effect of illumination differences.'
X = X / 255.0

'reshaping the dataset to fit standard of a 4D tensor of shape [mini-batch size, height = 28px, width = 28px, channels = 1 due to grayscale].'
X = X.values.reshape(-1,28,28,1)

'categorical conversion of label'
y = to_categorical(y, num_classes = 14)

'90% Training and 10% Validation split'
random_seed = 2
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1 , random_state = random_seed, stratify = y)


#-------------------------------

#~~~2. Model~~~
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Lambda
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from tensorflow import keras
import tensorflow as tf

''
model = Sequential()

'This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.'
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = "Same", activation = "relu", input_shape = (28, 28, 1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = "Same", activation = "relu"))
'MaxPool2D'
model.add(MaxPool2D(pool_size = (2,2)))
'Dropout'
model.add(Dropout(0.25))

#layer 2
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = "Same", activation = "relu"))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = "Same", activation = "relu"))
#model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

'Flatten'
model.add(Flatten())
'Dense'
model.add(Dense(256, activation = "relu"))
'Dropout'
model.add(Dropout(0.25))
'out'
model.add(Dense(14, activation = "softmax"))


##Set the optimizer and annealer

# defining optimizer
optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay=0.0 )
# compiling model
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
# annealer for learning rate
learning_rate_reduction = ReduceLROnPlateau(monitor = "val_accuracy",
                                            patience = 3,
                                            verbose = 1,
                                            factor = 0.5,
                                            min_lr = 0.0001)


##Data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

# model fitting
epochs = 5 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86


history = model.fit_generator(
                                datagen.flow(X_train,y_train, batch_size=batch_size),
                                epochs = epochs, #An epoch is an iteration over the entire x and y data provided
                                validation_data = (X_val,y_val), #Data on which to evaluate the loss and any model metrics at the end of each epoch. 
                                verbose = 1, #output
                                steps_per_epoch=X_train.shape[0] // batch_size,  # Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch.
                                callbacks=[learning_rate_reduction]                            
                              )


#~~~4. Saving Model~~~
model.save(r"C:\Users\20044\Desktop\Digital Recognizer\Trained_Model\model_v2.h5")

model = keras.models.load_model(r"C:\Users\20044\Desktop\Digital Recognizer\Trained_Model\model_v1.h5")

#~~~5. Prediction~~~
from PIL import Image
from itertools import groupby

'loading image'
image = Image.open(r"C:\Users\20044\Desktop\Digital Recognizer\Samples\test1.png").convert("L")


'resizing to 28 height pixels while maintaining the aspect ratio'
w = image.size[0]
h = image.size[1]
r = w / h # aspect ratio
new_w = int(r * 28)
new_h = 28
image = image.resize((new_w, new_h))

'converting to a numpy array'
image_arr = np.array(image)

'inverting'
image_arr = 255 - image_arr

'rescaling the image'
image_arr = image_arr / 255.0

'grouping by a non zero columns'
m = image_arr.any(0)
out = [image_arr[:,[*g]] for k, g in groupby(np.arange(len(m)), lambda x: m[x] != 0) if k]


#iterating through the split digits

'getting the number length'
num_len = len(out)

'creating a empty list to store all digits'
digit_list = []

for x in range(0, num_len):

    img = out[x]
    
    'adding zero columns to make it 28 pixels'
    width = img.shape[1]
    filler = (image_arr.shape[0] - width) / 2 #referecing the shape of the original image to make a square
    
    if filler.is_integer() == False:
        filler_l = int(filler)
        filler_r = int(filler) + 1
    else:
        filler_l = int(filler)
        filler_r = int(filler)
    
    'left'
    arr_l = np.zeros((image_arr.shape[0], filler_l))
    'right'
    arr_r = np.zeros((image_arr.shape[0], filler_r))
    
    'concat'
    help_ = np.concatenate((arr_l, img), axis= 1)
    help_ = np.concatenate((help_, arr_r), axis= 1)
    
    'resize array 2d to 3d'
    help_.resize(28, 28, 1)
    
    'converting to image array standards'
    final_arr = help_
    
    'combining all digits'
    digit_list.append(final_arr)


'converting to a numpy array'
digit_array = np.array(digit_list)


#prediction using the model
'model prediction'
ans_digit =  model.predict(digit_array)

ans_digit = np.argmax(ans_digit,axis = 1)

#####################################################################
def math_operation(arr):
    
    op = {10, 11, 12, 13}
    m_exp = []
    temp = []
    
    'creating a list separating all elements'
    for item in arr:
        if item not in op:
            temp.append(item)
        else:
            m_exp.append(temp)
            m_exp.append(item)
            temp = []
    if temp:
        m_exp.append(temp)

    
    'converting the digits to numbers'
    i = 0
    num = 0
    for item in m_exp:
        if type(item) == list:
            num_len = len(item)
            for digit in item:
                num_len = num_len - 1
                num = num + ((10 ** num_len) * digit)
            m_exp[i] = str(num)
            num = 0
            i = i + 1
        else:
            m_exp[i] = str(item)
            m_exp[i] = m_exp[i].replace("10","/")
            m_exp[i] = m_exp[i].replace("11","+")
            m_exp[i] = m_exp[i].replace("12","-")
            m_exp[i] = m_exp[i].replace("13","*")
            
            i = i + 1
    
    'a string'
    m_exp_str = m_exp
    
    'joining the list of strings'
    separator = ' '
    m_exp_str = separator.join(m_exp_str)
    
    
    'getting the answer'
    answer = eval(m_exp_str)
    
    return (m_exp_str, answer)
#####################################################################

#combine to make a number
m_exp_str, answer = math_operation(ans_digit)

equation  = m_exp_str + " = " + str(answer)

print(equation)




