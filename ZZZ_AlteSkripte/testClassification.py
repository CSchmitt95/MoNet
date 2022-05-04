import tensorflow as tf
import pandas as pd

model = tf.keras.models.load_model('Results/999_Test/Models/Gürtel')
labels = []

with open('Results/999_Test/labels.txt') as label_txt:
    lables = label_txt.readlines()
    
print(labels)

df = pd.read_csv("Data/TestData/classification_recordings.csv")

predictions_onehot = model.predict(df)

print(predictions_onehot)
