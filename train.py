import os
from traceback import print_tb
import tensorflow as tf
import cProfile
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

#from scipy.special import softmax
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from matplotlib import rcParams

def trainModel(model, model_filename):
    callbacks = []
    #callback1 = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=50)
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=500))
    #callback3 = tf.keras.callbacks.EarlyStopping(monitor='precision', patience=50)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks, class_weight=class_weights)
    
    rcParams['figure.figsize'] = (18, 8)
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False

    x = len(history.history['accuracy'])+1

    plt.plot(
        np.arange(1, x), 
        history.history['loss'], label='Loss'
    )
    plt.plot(
        np.arange(1, x), 
        history.history['accuracy'], label='Accuracy'
    )
    plt.plot(
        np.arange(1, x), 
        history.history['precision'], label='Precision'
    )
    plt.plot(
        np.arange(1, x), 
        history.history['recall'], label='Recall'
    )
    plt.title('Evaluation metrics', size=20)
    plt.xlabel('Epoch', size=14)
    plt.ylim(0, 1.5)
    plt.legend()



    #print(confusion_matrix)

    #loss, acc = model.evaluate(X_test, y_test, verbose=2)
    #print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

    model.save(outputpath + "models/"+ filename + "_"  + model_filename)
    plt.savefig( outputpath + "graphs/"+ filename +"_" + model_filename + "_training.png", bbox_inches='tight')
    plt.clf()

    predictions_onehot = model.predict(X_test)
    #predictions_onehot = np.around(predictions)
    y_test_onehot = y_test
    y_test_cm = y_test_onehot.values.argmax(axis=1)
    y_pred_cm = predictions_onehot.argmax(axis=1)
    cm = confusion_matrix(y_test_cm, y_pred_cm)
    
    cm_df = pd.DataFrame(cm, index = ["Gehen", "Stehen", "Stolpern"], columns = ["Gehen", "Stehen", "Stolpern"])
    
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.savefig( outputpath + "graphs/"+ filename + "_" + model_filename + "_confusion.png", bbox_inches='tight')
    plt.clf()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Configuration options
outputpath = "Results/Spaziergang/"
filepath = "ExampleData/output/"
filename = "SensorB"
file = filepath + filename
random_state=42
validation_split = 0.1
test_split = 0.1


df_train = pd.read_csv(file +".csv")

df_train['Gehen'] = [
    1 if MovementName == "Gehen" else 0 for MovementName in df_train['MovementName']
]
df_train['Stehen'] = [
    1 if MovementName == "Stehen" else 0 for MovementName in df_train['MovementName']
]
df_train['Stolpern'] = [
    1 if MovementName == "Stolpern" else 0 for MovementName in df_train['MovementName']
]

num_total = len(df_train)
num_gehen = df_train['Gehen'].sum()
num_stehen = df_train['Stehen'].sum()
num_stolpern = df_train['Stolpern'].sum()
class_weights = {
    0 : (1 / num_gehen) * (num_total / 3.0), 
    1 : (1 / num_stehen) * (num_total / 3.0), 
    2 : (1 / num_stolpern) * (num_total / 3.0), 
}

print(class_weights)

y_train = df_train[['Gehen','Stehen','Stolpern']]
X_train = df_train.drop(['MovementName', 'Gehen', 'Stehen', 'Stolpern'], axis=1)


df_test = pd.read_csv( file + "_Test.csv")

df_test['Gehen'] = [
    1 if MovementName == "Gehen" else 0 for MovementName in df_test['MovementName']
]
df_test['Stehen'] = [
    1 if MovementName == "Stehen" else 0 for MovementName in df_test['MovementName']
]
df_test['Stolpern'] = [
    1 if MovementName == "Stolpern" else 0 for MovementName in df_test['MovementName']
]

y_test = df_test[['Gehen','Stehen','Stolpern']]
X_test = df_test.drop(['MovementName', 'Gehen', 'Stehen', 'Stolpern'], axis=1)


'''X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=test_split, random_state=random_state
)'''

print("Daten geladen... " + str(len(X_train)) + " Trainingsdaten und " + str(len(X_test)) + " Testdaten\nStarten?")

learning_rate_Adam = 0.0001
learning_rate_SGD = 0.003
epochs = 3600
epochs_per_iteration = 100
batch_size = 1500


tf.random.set_seed(42)

hugenet = tf.keras.Sequential([
    tf.keras.layers.Dense(1500, activation='relu'),
    tf.keras.layers.Dense(3000, activation='relu'),
    tf.keras.layers.Dense(3000, activation='relu'),
    tf.keras.layers.Dense(1500, activation='relu'),
    tf.keras.layers.Dense(786, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid')
])

hugenet.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate_SGD),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

bignet = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid')
])

bignet.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate_SGD),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

tinynet = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid')
])

tinynet.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate_SGD),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

trainModel(hugenet,"BigNet")
#trainModel(bignet,"MediumNet")
#trainModel(tinynet,"SmallNet")