import TrainNetworkUtils
import os
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import VisualizationUtil
from shutil import move
from TrainNetworkUtils import saveHistoryGraph
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


#Konstanten
CURRENT_NAME = "013_Diff_Null"
RESULT_DIR = "Results/" + CURRENT_NAME + "/"
VIS_DIR = RESULT_DIR + "DataVis/"
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
Path(VIS_DIR).mkdir(parents=True, exist_ok=True)


#Wir wollen mit den Daten aller Sensoren trainieren...
for sensorname in os.listdir(DATA_PATH):
    movements = []
    if sensorname.endswith(".csv"):
        print("Lese Datei " + sensorname + "...")
        file = os.path.join(DATA_PATH, sensorname)
        df = pd.read_csv(file, dtype=np.float32)

        # Raute aus erstem Label entfernen...
        first_label = df.columns.values.tolist()[0]
        df.rename({first_label : first_label[2:]}, axis=1, inplace=True)


        spalten = len(df.columns)   
        print(str(spalten) + " splaten... also..." + str (spalten % 1500) + " movements") 
        movementcount = spalten % 1500
        movements = df.columns.values.tolist()[0:movementcount]
        print("Movements: " + str(movements))
        #Für Alle Movements eine Beispielvisualisierung Abspeichern
        for movement in movements:
                movement_samples = df.loc[df[movement] == 1.0]
                random_sample = movement_samples.sample(n=1)
                VisualizationUtil.prepare(movement, random_sample.values.tolist()[0], 0)
                plt.savefig(VIS_DIR + "/" + sensorname[:-4] + "_" + movement + ".png" )    
                plt.clf()

        sensorname = sensorname[:-4]
        TrainNetworkUtils.saveDistributionGraph32(df, movements, GRAPHS_DIR + sensorname + "_distribution_before.png" )

        with open(RESULT_DIR + "labels.txt", "w") as labels:
            for movement in movements:
                labels.write(movement + "\n")

        X = df.drop(movements, axis=1).astype(np.float32)
        Y = df[movements]
        print(Y)
        #Split in Training und Test Daten
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.40, random_state=42)

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
        callbacks.append(tf.keras.callbacks.EarlyStopping(min_delta=0.02, monitor='loss', patience=500))

        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, validation_split = 0.1)
        
        TrainNetworkUtils.writeReport(LEARNING_RATE, EPOCHS, BATCH_SIZE, model, RESULT_DIR + "Parameters.txt", movements)


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
