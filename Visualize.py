import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "Data/TestData/"

def visualize(title, data, counter):
    length = len(data)/4
    plt.figure(counter, figsize=(18,8))
    plt.xlabel('Zeitverlauf', size=14)
    plt.plot(
        np.arange(1, length), 
        data[0:-4:4], 
        label='w'
    )
    plt.plot(
        np.arange(1, length),  
        data[1:-3:4],
        label='x'
    )
    plt.plot(
        np.arange(1, length),  
        data[2:-2:4],
        label='y'
    )
    plt.plot(
        np.arange(1, length),  
        data[3:-1:4],
        label='z'
    )
    plt.title(title, size=20)
    #plt.ylim(bottom=-100,top=100)
    plt.legend() 


def showVisualizedWindow(movement_name, window, counter):
    length = len(window)/4
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

'''
for filename in os.listdir(DATA_PATH):
    if filename.endswith(".csv"):
        print("Lese Datei " + filename + "...")
        file = os.path.join(DATA_PATH, filename)
        df = pd.read_csv(file, dtype=np.float32, header=None)
        df_list = df.values.tolist()
        #datasets.update({sensorname[:-4] : df_training})
        #print("Movements: " + str(movements))

        counter = 0
        for line in df_list:
            #name = line.iat[0,0]
            #window = line.drop('MovementName', axis=1)
            #window = window.iloc[0].tolist()
            #print("Line: " + str(line))
            showVisualizedWindow("Test",line, counter)
            counter = counter + 1
        plt.show()
'''