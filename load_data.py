# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 16:55:34 2022

@author: Yushan
"""
import os
import cv2
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from keras.utils.np_utils import to_categorical


from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

from models import model, acc_loss , img_augumentation, img_aug_gen


def load_data():
	train = pd.read_csv('Dataset/sign_mnist_train/sign_mnist_train.csv')
	test = pd.read_csv('Dataset/sign_mnist_test/sign_mnist_test.csv')
	
	return train, test
	
def img_process(train, test):	
	if not os.path.exists('train'):
		 os.mkdir('train')
	if not os.path.exists('test'):
	    os.mkdir('test')
	
	for i, row in enumerate(train.to_numpy()):
	    label = row[0]
	    data = row[1:]
	    data = data.reshape((28,28))
	    if not os.path.exists(f'train/{label}'):
	        os.mkdir(f'train/{label}')
	    cv2.imwrite(f'train/{label}/{i}.jpeg', data)
	    
	for i, row in enumerate(test.to_numpy()):
	    label = row[0]
	    data = row[1:]
	    data = data.reshape((28,28))
	    if not os.path.exists(f'test/{label}'):
	        os.mkdir(f'test/{label}')
	    cv2.imwrite(f'test/{label}/{i}.jpeg', data)
		
def normalize(train, test):
    y_train = train['label'].values
    x_train = train.drop(columns = ['label']).to_numpy().reshape((
        train.shape[0], 28, 28,1)).astype('float64')/255.0
    
    y_test = test['label'].values
    x_test = test.drop(columns = ['label']).to_numpy().reshape((
        test.shape[0], 28,28,1)).astype('float64')/255.0
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    return x_train, y_train, x_test, y_test

def img_to_df(img_path):
    IMG_DIR = img_path
    
    for img in os.listdir(IMG_DIR):
        img_array = cv2.imread(os.path.join(IMG_DIR,img), cv2.IMREAD_GRAYSCALE)
        # print(type(img_array))
        # print(img_array)
        
        img_pil = Image.fromarray(img_array)
        img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))
    
        img_array = (img_28x28.flatten())
    
        img_array  = img_array.reshape(-1,1).T
    
        # print(type(img_array))
        
        # print(img_array.size)
    
        with open('img_train.csv', 'ab') as f:
            np.savetxt(f, img_array, delimiter=",")
    
def classify(model, img_path='images'):
    img_to_df(img_path)
    test_df = pd.read_csv('img_train.csv', error_bad_lines=False)
    norm_test = test_df.to_numpy().reshape((test_df.shape[0], 28, 28,1)).astype('float64')/255.0
    
    pred = model.predict(norm_test)
    pred = np.argmax(pred, axis=1)
    
    return pred
    
if __name__ == '__main__':
    train, test = load_data()
    # img_process(train, test)
    x_train, y_train, x_test, y_test = normalize(train, test)
    n_labels = y_train.shape[1]
    
    # img_gen = img_augumentation()
    
    # out_dir = 'train'
    # train_gen = img_aug_gen(img_gen, out_dir)
    # out_dir = 'test'
    # test_gen = img_aug_gen(img_gen, out_dir)
    
    model = model()
    print(model.summary())
    
    checkpoint_path = 'checkpoint/cp.ckpt'
    checkpoint_dir = os.path.join(checkpoint_path)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)  
        
    history = model.fit(
        x_train, y_train,
        epochs = 22,
        validation_data = (x_test, y_test),
        callbacks=[cp_callback])
    
    # model.load_weights(checkpoint_path)
    
    
    
    
    


