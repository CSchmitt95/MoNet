import TrainNetworkUtils
import os
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from shutil import move
from TrainNetworkUtils import saveHistoryGraph
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


#Konstanten
CURRENT_NAME = "004_NeueDifferenzierung"
RESULT_DIR = "Results/" + CURRENT_NAME + "/"
GRAPHS_DIR = RESULT_DIR + "Graphs/"
MODELS_DIR = RESULT_DIR + "Models/"


DATA_PATH = "Data/TrainingData/"
LEARNING_RATE = 0.003
EPOCHS = 7000
BATCH_SIZE = 1000

#Variablen
datasets = {}

#initialisieren
Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
Path(GRAPHS_DIR).mkdir(parents=True, exist_ok=True)
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

#Wir wollen mit den Daten aller Sensoren trainieren...
for sensorname in os.listdir(DATA_PATH):
    if sensorname.endswith(".csv"):
        print("Lese Datei " + sensorname + "...")
        file = os.path.join(DATA_PATH, sensorname)
        df = pd.read_csv(file)
        #datasets.update({sensorname[:-4] : df_training})
        movements = df['MovementName'].unique()
        print("Movements: " + str(movements))
    
    sensorname = sensorname[:-4]
    TrainNetworkUtils.saveDistributionGraph(df, GRAPHS_DIR + sensorname + "_distribution_before.png" )

    print("One-Hot Aufbereitung für " + sensorname + "...")
    for movement in movements:
        df[movement] = [
            1 if MovementName == movement else 0 for MovementName in df['MovementName']
        ]


    droplist = ["MovementName"]
    droplist.extend(movements)

    X = df.drop(droplist, axis=1).astype(np.float32)
    Y = df[movements]

    #Split in Training und Test Daten
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    model = tf.keras.Sequential([ 
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
    ])


    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1500))

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, validation_split = 0.1)
    
    TrainNetworkUtils.writeReport(LEARNING_RATE, EPOCHS, BATCH_SIZE, model, RESULT_DIR + "Parameters.txt")


    model.save(MODELS_DIR + sensorname)

    TrainNetworkUtils.saveHistoryGraph(history, GRAPHS_DIR + sensorname + "_history.png")

    predictions_onehot = model.predict(X_test)
    y_test_onehot = y_test
    y_test_cm = y_test_onehot.values.argmax(axis=1)
    y_pred_cm = predictions_onehot.argmax(axis=1)
    cm = confusion_matrix(y_test_cm, y_pred_cm)
    cm_df = pd.DataFrame(cm, index = movements, columns =movements)
    TrainNetworkUtils.saveConfusionMatrix(cm_df, GRAPHS_DIR + sensorname + "_confusion.png")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open (MODELS_DIR + sensorname + ".tflite", 'wb') as f:
        f.write(tflite_model)
