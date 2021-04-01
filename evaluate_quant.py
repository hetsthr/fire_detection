import tensorflow as tf
import time
from os import listdir
from os.path import isfile, join
import numpy as np

img_dir = 'fire_data/test/'
MODEL_DIR = 'TRAINED_MODELS'
MODEL_NAME = 'mobilenet'
CHANNELS = 1
IMG_SIZE = 32

if CHANNELS == 1:
    MODEL_FOLDER = MODEL_NAME + '_gray'
    MODEL_NAME = MODEL_NAME + '_gray'
else:
    MODEL_FOLDER = MODEL_NAME + '_rgb'

MODEL_PATH = MODEL_DIR+'/'+MODEL_FOLDER+'/'+MODEL_NAME+'_'+str(IMG_SIZE)+'/'+MODEL_NAME+'_'+str(IMG_SIZE)+'_quant.tflite'

interpreter = tf.lite.Interpreter(model_path = MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
print(input_shape)

dataset_list = tf.data.Dataset.list_files(img_dir + '*')
avg = 0
for i in range(10):
    image = next(iter(dataset_list))
    print(image, end=' ')
    image = tf.io.read_file(image)
    image = tf.io.decode_png(image, channels=CHANNELS)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    input_im = tf.expand_dims(image, 0)

    input_data = np.array(input_im, dtype=np.int8)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    start = time.time()
    interpreter.invoke()
    stop = time.time()
    
    avg = avg + (stop-start)

    output_data = interpreter.get_tensor(output_details[0]['index'])
    res = np.argmax(output_data, axis = 1)
    print(res)

print('Latency: ', avg/10)
