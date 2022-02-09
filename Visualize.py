import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "Data/TrainingData/"


def showVisualizedWindow(movement_name, window, counter):
    length = len(window)/4
    w = window[0:-4:4], 
    plt.figure(counter, figsize=(18,8))
    plt.plot(
        np.arange(1, length), 
        window[0:-4:4], 
        label='w'
    )
    plt.plot(
        np.arange(1, length),  
        window[1:-3:4],
        label='x'
    )
    plt.plot(
        np.arange(1, length),  
        window[2:-2:4],
        label='y'
    )
    plt.plot(
        np.arange(1, length),  
        window[3:-1:4],
        label='z'
    )
    plt.title(movement_name, size=20)
    plt.xlabel('Zeitverlauf', size=14)
    plt.ylim(-1, 1)
    plt.legend() 

for sensorname in os.listdir(DATA_PATH):
    if sensorname.endswith(".csv"):
        print("Lese Datei " + sensorname + "...")
        file = os.path.join(DATA_PATH, sensorname)
        df = pd.read_csv(file)
        #datasets.update({sensorname[:-4] : df_training})
        movements = df['MovementName'].unique()
        print("Movements: " + str(movements))

        counter = 0
        for movement in movements:
            lines = df.loc[df['MovementName'] == movement]
            line = lines.sample(n=1)
            name = line.iat[0,0]
            window = line.drop('MovementName', axis=1)
            window = window.iloc[0].tolist()
            showVisualizedWindow(name,window, counter)
            counter = counter + 1
        plt.show()

'''
        while True:
            line = df.sample(n=1)
            name = df.iat[0,0] #line['MovementName']
            window = line.drop('MovementName', axis=1)
            window = window.iloc[0].tolist()
            showVisualizedWindow(name,window)


        for movement in movements:
            window = df[movement].sample(n=1)
            window.drop("MovementName", axis=1)
            showVisualizedWindow(window)
'''