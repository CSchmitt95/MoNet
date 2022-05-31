import os
from shutil import move
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import VisualizationUtil

DATA_PATH = "Data/PreprocessedData/"

for filename in os.listdir(DATA_PATH):
    if filename.endswith(".csv"):
        print("Lese Datei " + filename + "...")
        file = os.path.join(DATA_PATH, filename)
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

        counter = 0
        for movement in movements:
            lines = df.loc[df[movement] == 1.0]
            line = lines.sample(n=1)
            VisualizationUtil.prepare(movement, line.values.tolist()[0], counter)
            counter += 1
            plt.savefig(DATA_PATH + "/" + filename[:-4] + "_" + movement + ".png" )    
        #VisualizationUtil.showGraphs()
        input()
