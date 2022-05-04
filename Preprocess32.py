INPUT_PATH = "Data/RecordedData/"
OUTPUT_PATH = "Data/ProcessedData/"

import os
import csv
import numpy as np
import pandas as pd
import PreprocessUtil
import pprint
import PreprocessConstants
import VisualizationUtil

from ast import operator
from pathlib import Path
from shutil import move
from operator import itemgetter
from matplotlib.pyplot import get
from progress.bar import Bar

#Liste, die alle CSV datein einliest.
complete_data = []

print("Bitte Datensatz Namen angeben")
DATASET = input()

print("Bitte Ausgabenamen angeben")
OUTPUTNAME = input()


Path(OUTPUT_PATH+"/"+OUTPUTNAME).mkdir(parents=True, exist_ok=True)


print("Einlesen:")

#Erst werden alle CSV Daten gelesen. Jede Zeile ist eine Reihe von Daten. Die noch zerteilt werden muss und gegebenenfalls mit dem anderen Sensor kombiniert werden muss.
directory = INPUT_PATH + DATASET + '/'
filecounter = 0
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filecounter = filecounter+1
        file = os.path.join(directory, filename)
        with open(file, newline='', encoding='utf-8') as input_file:
            reader = csv.reader(input_file)
            data = list(reader)
            complete_data.extend(data[1:])
            print("Datei " + filename + " gefunden... " + str(len(data)) + " Einträge.")

print()


#Jetzt Zählen wir wie viele Sensoren und wie viele Bewegungen aufgenommen wurden
#Der Header ist MovementName,SensorName,Record_id,w0,x0,y0 z0, ... , wn, xn, yn, zn
#Dementsprechend:  line[0]  ,  line[1] , [line2]
print("Übersicht:")
print("Insgesamt " + str(filecounter) + " CSVs gelesen... " + str(len(complete_data)) + " Einträge.")
sensors = []
movements = []
for line in complete_data:
    if line[0] not in movements:
        movements.append(line[0])
    if line[1] not in sensors:
        sensors.append(line[1])

sensorcount = len(sensors)
movementcount = len(movements)
print(str(len(sensors)) + " Sensoren: " + str(sensors))
print(str(len(movements)) + " Bewegungen: " + str(movements)) 
print()


#Konsistenzcheck
number_of_recordings = len(complete_data)/sensorcount
if number_of_recordings%1 != 0.0:
    print("FEHLER:")
    print("Das hier sollte eine Ganze zahl sein: " + str(number_of_recordings) + " ... " + str())
    exit()

#print("Bilde Differenzquaternionen...")
#PreprocessUtil.differentiateCorrectly(complete_data)

windowed_data = {}
for i in range (0, sensorcount):
    windowed_data.update({sensors[i] : np.array([], dtype=np.float32).reshape(0, movementcount + PreprocessConstants.QUATERNIONS_PER_WINDOW)})
print()

print("Generiere Windows...")
with Bar("Aufnahmen", max=number_of_recordings) as bar:
    #Für Jedes Sensorpaar
    for i in range(0, int(number_of_recordings)):
        movement_name = complete_data[i*sensorcount][0]
        #Für jeden Sensor im Datenpaar
        for j in range (0, sensorcount):
            line = complete_data[i*sensorcount + j]
            if line[0] != movement_name:
                print("------>Fehlerhafte Datenformatierung!<------")
                print("Movement Name is: " + movement_name)
                exit()
            sensor_name = line[1]
            current_windows = windowed_data.get(sensor_name)
            new_windows = PreprocessUtil.getWindowsFromLine32(line, movements) 
            current_windows = np.concatenate((current_windows, new_windows), axis=0)
            windowed_data.update({sensor_name : current_windows })
        bar.next()
print()

print("Nulle Windows an erster Quaternion...")
for sensor in sensors:
    windows = windowed_data.get(sensor)
    normalized_windows = PreprocessUtil.normalizeWindows32(windows)
    windowed_data.update({ sensor: normalized_windows})
print()

#Kombiniere Alle Sensordaten in Kombo-Klasse
print("Kombiniere windows...")
combined_windows, kombo_name = PreprocessUtil.getCombinedWindowsOf32(windowed_data)
windowed_data.update({kombo_name : combined_windows})
sensors.append(kombo_name)
print("")

#Daten in die CSVs schreiben.
print("Schreibe CSVs...")
for sensor in sensors:
    writing_windows = windowed_data.get(sensor)
    PreprocessUtil.writeWindowsToFile32(windowed_data.get(sensor), OUTPUT_PATH + "/" + OUTPUTNAME + "/" + sensor + ".csv", movements)
