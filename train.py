import os
import tensorflow as tf
import cProfile
import numpy as np
import pandas as pd

from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

#from scipy.special import softmax
from sklearn.metrics import multilabel_confusion_matrix

import matplotlib.pyplot as plt
from matplotlib import rcParams

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Configuration options
file = "ExampleData/output/SensorB"
random_state=42
validation_split = 0.1
test_split = 0.1

learning_rate = 0.02
epochs = 5000
batch_size = 500


df_train = pd.read_csv(file +".csv")

df_train['Gehen'] = [
    1 if MovementName == "Gehen" else 0 for MovementName in df_train['MovementName']
]
df_train['huepfen'] = [
    1 if MovementName == "huepfen" else 0 for MovementName in df_train['MovementName']
]
df_train['stehen'] = [
    1 if MovementName == "stehen" else 0 for MovementName in df_train['MovementName']
]

y_train = df_train[['Gehen','stehen','huepfen']]
X_train = df_train.drop(['MovementName', 'Gehen', 'stehen', 'huepfen'], axis=1)

'''X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=test_split, random_state=random_state
)'''


df_test = pd.read_csv( file + "_Test.csv")

df_test['Gehen'] = [
    1 if MovementName == "Gehen" else 0 for MovementName in df_test['MovementName']
]
df_test['huepfen'] = [
    1 if MovementName == "huepfen" else 0 for MovementName in df_test['MovementName']
]
df_test['stehen'] = [
    1 if MovementName == "stehen" else 0 for MovementName in df_test['MovementName']
]

y_test = df_test[['Gehen','stehen','huepfen']]
X_test = df_test.drop(['MovementName', 'Gehen', 'stehen', 'huepfen'], axis=1)


print("Daten geladen... " + str(len(X_train)) + " Trainingsdaten und " + str(len(X_test)) + " Testdaten\nStarten?")
input()


tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

plt.plot(
    np.arange(1, epochs+1), 
    history.history['loss'], label='Loss'
)
plt.plot(
    np.arange(1, epochs+1), 
    history.history['accuracy'], label='Accuracy'
)
plt.plot(
    np.arange(1, epochs+1), 
    history.history['precision'], label='Precision'
)
plt.plot(
    np.arange(1, epochs+1), 
    history.history['recall'], label='Recall'
)
plt.title('Evaluation metrics', size=20)
plt.xlabel('Epoch', size=14)
plt.legend()

plt.show()


predictions = model.predict(X_test)
predictions = np.around(predictions)
print(multilabel_confusion_matrix(y_test, predictions))
