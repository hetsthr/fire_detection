from keras.models import load_model
import tensorflow as tf
import os
import numpy as np
from os import listdir
from os.path import isfile, join
import time

labels_dict = {"[0]" : "Fire", "[1]" : "Non-Fire"}

img_dir = 'fire_data/test/'
dataset_list = tf.data.Dataset.list_files(img_dir + "*")

MODEL_DIR = './TRAINED_MODELS'
MODEL_NAME = 'cnn'
CHANNELS = 3
IMG_SIZE = 32

if CHANNELS == 1:
	MODEL_FOLDER = MODEL_NAME + "_gray"
	MODEL_NAME = MODEL_NAME + "_gray"
else:
	MODEL_FOLDER = MODEL_NAME + "_rgb"

MODEL_PATH = MODEL_DIR+"/"+MODEL_FOLDER+"/"+MODEL_NAME+"_"+str(IMG_SIZE)+"/"+MODEL_NAME+"_"+str(IMG_SIZE)+'.h5'

classifier = load_model(MODEL_PATH)

#correct = 0
avg = 0
for i in range(0,10):
	image = next(iter(dataset_list))
	print(image, end=' ')
	image = tf.io.read_file(image)
	image = tf.io.decode_jpeg(image, channels=CHANNELS)
	image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
	image = tf.expand_dims(image, 0)

	start = time.time()
	res = np.argmax(classifier.predict(image, 1, verbose = 0), axis = 1)
	stop = time.time()
	avg = avg + (stop-start)
	print("Prediction: ", labels_dict[str(res)])

print("Latency (k): ", avg/10)
