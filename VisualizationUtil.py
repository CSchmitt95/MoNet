import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def immediatelyShowGraph(title, window, counter):
    prepare(title, window, counter)
    showGraphs()

def showGraphs():
    plt.show()

def prepare(title, window, counter):
    offset = len(window) % 1500
    plotQuaternions(title, window[offset:], counter)

def plotSensorData(data, counter):
    movement_name = data[0]
    sensor_name = data[1]
    record_id = data[2]

    plotQuaternions(movement_name+ " ("+ sensor_name +")", data[3:], counter)

def plotQuaternions(title, window, counter):
    length = (len(window))/4
    plt.figure(counter, figsize=(18,8))
    plt.xlabel('Zeitverlauf', size=14)
    plt.xlim = (0, length)
    plt.plot(
        np.arange(0, len(window[0::4])), 
        window[0::4], 
        label='w'
    )
    plt.plot(
        np.arange(0, len(window[1::4])),  
        window[1::4],
        label='x'
    )
    plt.plot(
        np.arange(0, len(window[2::4])),  
        window[2::4],
        label='y'
    )
    plt.plot(
        np.arange(0, len(window[3::4])),  
        window[3::4],
        label='z'
    )
    plt.title(title, size=20)
    plt.ylim(bottom=-1.1,top=1.1)
    plt.legend()


def visualizeTrainingData(trainingdata_path, output_path):
    for filename in os.listdir(trainingdata_path):
        if filename.endswith(".csv"):
            print("Lese Datei " + filename + "...")
            file = os.path.join(trainingdata_path, filename)
            df = pd.read_csv(file)
            df_list = df.values.tolist()

            #Bewegungen finden
            spalten = len(df.columns)   
            movementcount = spalten % 1500
            movements = df.columns.values.tolist()[0:movementcount]
            #Raute aus erstem Namen entfernen
            first_label = df.columns.values.tolist()[0]
            movements[0] = first_label[2:]
            df.rename({first_label : first_label[2:]}, axis=1, inplace=True)
            
            print("Movements: " + str(movements))

            for movement in movements:
                lines = df.loc[df[movement] == 1.0]
                line = lines.sample(n=1)
                prepare(movement, line.values.tolist()[0], 0)
                plt.savefig(output_path + "/" + filename[:-4] + "_" + movement + ".png" )    
                plt.clf()
