import shutil
import os
import sys
from flask import Flask, render_template, request
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt

from utile import daugman, DiscreteCosineTransform, msg_encrypt, find_iris

from rubikencryptor.rubikencryptor import RubikCubeCrypto
from PIL import Image

key = 'key.txt'

# global b
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/encrypt', methods=['GET', 'POST'])
def encrypt():
    if request.method == 'POST':
        fileName=request.form['filename1']
        fileName2=request.form['filename2']
        fileName3=request.form['filename3']
        
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'iris-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            path = "static/iris/"+fileName2
            img_num = fileName2.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
            np.save('verify_data1.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log1')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        str_label1=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 1:
                str_label1 = 'Fake'
            elif np.argmax(model_out) == 0:
                str_label1 = 'Live'

        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'fingerprint-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            path = "static/fingerprint/"+fileName3
            img_num = fileName3.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
            np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        str_label=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                str_label = 'Fake'
            elif np.argmax(model_out) == 1:
                str_label = 'Live'

        if str_label == 'Live' and str_label1 == 'Live':
            input_image = "static/original/"+fileName
            output_image = "static/encrypted/"+fileName

            input_image = Image.open(input_image)
            rubixCrypto = RubikCubeCrypto(input_image)
            
            encrypted_image = rubixCrypto.encrypt(alpha=8, iter_max=10, key_filename=key)
            encrypted_image.save(output_image)

            import random
            otp2 = random.randint(1000,9999)
            otp2 = str(otp2)
            print(otp2)

            import telepot
            bot = telepot.Bot("5629397463:AAE1lrJJquL-MLIZeaMzlvzASLbAuqRgPzI")
            bot.sendMessage("1850422634", str(otp2))



            return render_template('home.html', result1 = fileName, otp2=otp2)
        else:
            return render_template('home.html', msg="fingerprint or iris does not match")
    return render_template('home.html')

@app.route('/verify1', methods=['GET', 'POST'])
def verify1():
    if request.method == 'POST':
        fileName=request.form['filename1']
        otp1=int(request.form['otp1'])
        otp2=int(request.form['otp2'])

        if otp1 == otp2:
            return render_template('home.html',  ImageDisplay1="static/original/"+fileName, ImageDisplay2="static/encrypted/"+fileName)
        else:
            return render_template('home.html', msg="Entered wrong otp")

        return render_template('home.html')
    return render_template('home.html')

@app.route('/decrypt', methods=['GET', 'POST'])
def decrypt():
    if request.method == 'POST':
        fileName1=request.form['filename1']
        fileName2=request.form['filename2']
        fileName3=request.form['filename3']

        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'iris-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            path = "static/iris/"+fileName2
            img_num = fileName2.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
            np.save('verify_data1.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log1')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        str_label1=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 1:
                str_label1 = 'Fake'
            elif np.argmax(model_out) == 0:
                str_label1 = 'Live'

        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'fingerprint-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            path = "static/fingerprint/"+fileName3
            img_num = fileName3.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
            np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        str_label=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                str_label = 'Fake'
            elif np.argmax(model_out) == 1:
                str_label = 'Live'

        if str_label == 'Live' and str_label1 == 'Live':
            input_image = "static/encrypted/"+fileName1
            output_image = "static/decrypted/"+fileName1
            
            input_image = Image.open(input_image)
            rubixCrypto = RubikCubeCrypto(input_image)
            
            decrypted_image = rubixCrypto.decrypt(key_filename=key)
            decrypted_image.save(output_image)

            import random
            otp2 = random.randint(1000,9999)
            otp2 = str(otp2)
            print(otp2)

            import telepot
            bot = telepot.Bot("5629397463:AAE1lrJJquL-MLIZeaMzlvzASLbAuqRgPzI")
            bot.sendMessage("1850422634", str(otp2))

            return render_template('home.html', result= fileName1, otp2=otp2)
        else:
            return render_template('home.html', msg="fingerprint or iris does not match")

    return render_template('home.html')

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        fileName1=request.form['filename1']
        otp1=int(request.form['otp1'])
        otp2=int(request.form['otp2'])

        if otp1 == otp2:
            return render_template('home.html', Image1="http://127.0.0.1:5000/static/encrypted/"+fileName1, Image2="http://127.0.0.1:5000/static/original/"+fileName1)
        else:
            return render_template('home.html', msg="Entered wrong otp")

        return render_template('home.html')
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
