import os
import tensorflow as tf
from tensorflow import keras


model = tf.lite.TFLiteConverter.from_saved_model("New/Results/002_MehrStolperDaten/Models/Gürtel")
tflite_model = model.convert()

with open ('New/Results/002_MehrStolperDaten/Models/Gürtel.tflite', 'wb') as f:
    f.write(tflite_model)