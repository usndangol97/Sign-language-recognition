# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 11:57:13 2022

@author: Yushan
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


def model():
    model = tf.keras.models.Sequential()        
    model.add(tf.keras.layers.Conv2D(64,(3,3),padding ='Same',activation = 'relu',input_shape=(28,28,1)))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(128,(3,3),padding ='same',activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(128,(3,3),padding ='same',activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(25, activation ='softmax'))   
    
    model.compile(
        optimizer= 'adam',
        loss='categorical_crossentropy',
        metrics = ['accuracy'])
    
    return model

def acc_loss(history):
    fig, axes = plt.subplots(nrows = 2, ncols = 1)
    axes[0].plot(history.history['accuracy'], label = 'accuracy')
    axes[0].plot(history.history['loss'], label = 'loss')
    axes[1].plot(history.history['val_accuracy'], label = 'val. accuracy')
    axes[1].plot(history.history['val_loss'], label = 'val. loss')
    
    
    axes[0].legend()
    axes[0].grid(True)
    axes[1].legend()
    axes[1].grid(True)
    plt.show()
    
def img_augumentation():
    img_gen = ImageDataGenerator(
        rescale = 1/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.25)
    
    return img_gen
    
def img_aug_gen(img_gen,out_dir):
    generator = img_gen.flow_from_directory(
        out_dir,
        class_mode = 'categorical',
        batch_size = 20,
        target_size = (28,28))
    
    return generator
    
    
    
    
    
    
    
