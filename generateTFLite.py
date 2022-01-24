import os
import tensorflow as tf
from tensorflow import keras


model = tf.lite.TFLiteConverter.from_saved_model("Results/Spaziergang_balanced/models/SensorA_bigNet_earlystop")
tflite_model = model.convert()

with open ('model.tflite', 'wb') as f:
    f.wirte(tflite_model)