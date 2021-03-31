from keras.models import load_model
import tensorflow as tf
data_dir = './fire_data'
img_dir = data_dir + '/test/'
CHANNELS = 1
IMG_SIZE = 96
def rep_data_gen():
    dataset_list = tf.data.Dataset.list_files(img_dir + '*')
    for i in range(100):
        image = next(iter(dataset_list))
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=CHANNELS)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = tf.expand_dims(image, 0)
        yield [image]

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('mobilenet_gray_96.h5')

tflite_float = converter.convert()

with open('mobilenet_gray_96_float.tflite', 'wb') as f:
    f.write(tflite_float)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = rep_data_gen

converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant = converter.convert()

with open('mobilenet_gray_96_quant.tflite', 'wb') as f:
    f.write(tflite_quant)
'''
!xxd -i cnn_32_quant.tflite > cnn_32.cc

REPLACE_TEXT = 'cnn_96_quant.tflite'.replace('/', '_').replace('.','_')
!sed -i 's/'{REPLACE_TEXT}'/g_model/g' 'cnn_96.cc'
!tail cnn_96.cc
'''
