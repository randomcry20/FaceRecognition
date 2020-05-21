#!/usr/bin/env python
# coding: utf-8

# ## Importing MobileNet


#Importing the pre-trained model along with pre-trained weigths
from keras.applications import MobileNet
model = MobileNet(weights='imagenet', include_top = False, input_shape=(224,224,3))
for (i,layer) in enumerate(model.layers):
    layer.trainable=False # FREEZING THE LAYERS!!!  
    #print(str(i) + " " + layer.__class__.__name__ + " " + str(layer.trainable))


# ## Addition of an Untrained FC layers


#DEFINING A FUNCTION WHICH WILL ADD FULLY CONNECTED LAYERS 
def newfc(base_model, categories):
    top_layer= base_model.output
    top_layer=Conv2D(filters=256, kernel_size=(2,2), activation='relu')(top_layer)
    #Convolution Layer Will Be Added After top_Layer
    top_layer = GlobalAveragePooling2D()(top_layer)
    top_layer= Dense(units=1024, activation='relu')(top_layer)
    top_layer= Dense(units=512, activation = 'relu')(top_layer)
    top_layer= Dense(units = 64, activation= 'relu')(top_layer)
    top_layer= Dense(categories, activation='softmax',)(top_layer) # Output Layer
    return top_layer

# Here, we will create our model.
# Right Now, Our FC Layers are untrained.
# We'll train
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
pred_categories = 3
FC_NEW= newfc(model, pred_categories)
new_model= Model(inputs = model.input, outputs=FC_NEW) 
# Model will combine the FC_NEW after model.input layer!! 

new_model.summary()


# AUGMENTATION
from keras.preprocessing.image import ImageDataGenerator

train_data_directory="C://Users//HP//Desktop//MLOps//Face//Train//"
validation_data_Directory= "C://Users//HP//Desktop//MLOps//Face//Validation//"

train_datagen= ImageDataGenerator(rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
validation_datagen= ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_data_directory,
                                                    target_size = (224,224),
                                                      class_mode='categorical',
                                                        batch_size=32) 

validation_generator = validation_datagen.flow_from_directory(validation_data_Directory, 
                                                             target_size=(224,224), 
                                                             class_mode = 'categorical', 
                                                              batch_size=32)

from keras.optimizers import RMSprop
new_model.compile(optimizer=RMSprop(learning_rate = 0.001), 
                  loss="categorical_crossentropy",
                    metrics=['accuracy'])
new_model.fit_generator(train_generator, 
                         epochs = 3,
                          validation_data = validation_generator,
                           steps_per_epoch = 100,
                            validation_steps = 25)                   


#  #  Prediction

# #   Prediction Of Some Randomly Generated Input Images 
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

face_dict = {"[0]": "Naman", "[1]": "Cat", "[2]": "Dog"}

face_dict_actual = {"Naman": "Naman", "dogs": "Dog", "cats": "Cat"}

def draw_test(name, pred, im):
    face = face_dict[str(pred)]
    cv2.putText(im, face, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow(name, im)
    
# THIS FUNCTION WILL GENERATE A RANDOM IMAGE  
def getRandomImage(path):
    # Generator will give a list which contains name of all the folders.
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path))) 
    random_folder_index = np.random.randint(0,len(folders))
    path_class = folders[random_folder_index]
    print("Correct Output - " + face_dict_actual[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    random_image = cv2.imread(file_path+"/"+image_name)
    return random_image 

for i in range(0,5):
    input_im = getRandomImage("Face/Validation/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, 
                                fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    # Prediction: Argmax will return the maximum argument
    res1 = np.argmax(new_model.predict(input_im, 1, verbose = 0), axis=1)

    
    # Show image with predicted class
    draw_test("Prediction", res1, input_original) 
    cv2.waitKey(0)

cv2.destroyAllWindows()

