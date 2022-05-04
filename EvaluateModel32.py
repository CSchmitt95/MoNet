import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from shutil import move
from TrainNetworkUtils import saveHistoryGraph
from pathlib import Path
import TrainNetworkUtils

from sklearn.metrics import confusion_matrix



MODEL_NAME = input("Welches Modell soll getestet werden?")
TEST_DATA = input("Welcher Datensatz soll verwendet werden?")

DATA_PATH = "Data/ProcessedData/"+ TEST_DATA + "/"

MODEL_DIR = "TrainingResults/" + MODEL_NAME + "/"
RESULT_ROOT = "EvaluationResults/" + MODEL_NAME + "/"
RESULT_DIR = RESULT_ROOT + "/" + TEST_DATA + "/"

#Mkdir für wichtige Ordner.
Path(RESULT_ROOT).mkdir(parents=True, exist_ok=True)
Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)

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
        
        X_test = df.drop(movements, axis=1).astype(np.float32)
        y_test = df[movements]

        model = keras.models.load_model(MODEL_DIR + "Models/" + sensorname[0:-4])
        
        predictions_onehot = model.predict(X_test)
        y_test_onehot = y_test
        y_test_cm = y_test_onehot.values.argmax(axis=1)
        y_pred_cm = predictions_onehot.argmax(axis=1)
        cm = confusion_matrix(y_test_cm, y_pred_cm)
        cm_df = pd.DataFrame(cm, index = movements, columns =movements)
        cm_df.to_csv(RESULT_DIR + sensorname[0:-4] + ".csv", index=True)

        #TrainNetworkUtils.saveConfusionMatrix(cm_df, RESULT_DIR + sensorname[0:-4] + "_confusion.png")

