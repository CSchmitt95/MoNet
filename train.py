import os
import tensorflow as tf
import cProfile
import numpy as np
import pandas as pd

from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

from scipy.special import softmax

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


df = pd.read_csv("ExampleData/output/SensorA.csv")
    

'''
df['Dancen'] = [
    1 if MovementName == "dancen" else 0 for MovementName in df['MovementName']
]

df['Dancen'] = [
    1 if MovementName == "dancen" else 0 for MovementName in df['MovementName']
]

df.drop('MovementName', axis=1, inplace=True)

print(df)

X = df.drop('Dancen', axis=1)
y = df['Dancen']
'''

# Configuration options
n_samples = (df.shape[0] - 1) 
n_features = (df.shape[1] - 1)
random_state=42
validation_split = 0.3
epochs = 100


df['Gehen'] = [
    1 if MovementName == "Gehen" else 0 for MovementName in df['MovementName']
]
df['huepfen'] = [
    1 if MovementName == "huepfen" else 0 for MovementName in df['MovementName']
]
df['stehen'] = [
    1 if MovementName == "stehen" else 0 for MovementName in df['MovementName']
]

y = df[['Gehen','stehen','huepfen']]
y.append = df['stehen']
y.append = df['huepfen']

X = df.drop(['MovementName', 'Gehen', 'stehen', 'huepfen'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=validation_split, random_state=random_state
)

#print(X_train)
#print(y_train)
#np.savetxt("test.csv", X_train, delimiter=',')
labels = y_test.columns.values


tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),    
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

history = model.fit(X_train, y_train, epochs=epochs)

import matplotlib.pyplot as plt
from matplotlib import rcParams

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

#print("Prediction:")
predictions = model.predict(X_test)
#print (predictions)


#print("Prediction Softmax")
predictions = softmax(predictions, axis=1)
#print (predictions)

#print("Prediction Final")
predictions = np.around(predictions)
#print (predictions)



print("y_test:")
print (y_test)


from sklearn.metrics import multilabel_confusion_matrix
print(multilabel_confusion_matrix(y_test, predictions))
