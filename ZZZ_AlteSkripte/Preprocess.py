DATASET = "KleinerSpaziergang"
OUTPUTPATH = "Data/TrainingData/"

from ast import operator
import os
import csv
import numpy as np
from shutil import move
from operator import itemgetter
from matplotlib.pyplot import get
import pandas as pd
import PreprocessUtil
from progress.bar import Bar
import pprint
from pathlib import Path

#Liste, die alle CSV datein einliest.
complete_data = []

Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True)

print("Einlesen:")

#Erst werden alle CSV Daten gelesen. Jede Zeile ist eine Reihe von Daten. Die noch zerteilt werden muss und gegebenenfalls mit dem anderen Sensor kombiniert werden muss.
directory = "Data/DataSets/" + DATASET + '/'
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
number_of_chunks = len(complete_data)/sensorcount
if number_of_chunks%1 != 0.0:
    print("FEHLER:")
    print("Das hier sollte eine Ganze zahl sein: " + str(number_of_chunks) + " ... " + str())
    exit()

print("Bilde Differenzquaternionen...")
PreprocessUtil.differentiateCorrectly(complete_data)

#Generiere Output Dictionary und passende Header
print("Generiere Output Dictionary")
print("Schreibe Header...")
windowed_data = {}
for i in range (0, sensorcount):
    windowed_data.update({sensors[i] : [PreprocessUtil.generateWindowHeader(0)]})
print()

#Fülle Output Dictionary
print("Schreibe Daten...")
with Bar("Aufnahmen", max=number_of_chunks) as bar:
    for i in range(0, int(number_of_chunks)):
        movement_name = complete_data[i*sensorcount][0]
        for j in range (0, sensorcount):
            line = complete_data[i*sensorcount + j]
            if line[0] != movement_name:
                print("------>Fehlerhafte Datenformatierung!<------")
                print("Movement Name is: " + movement_name)
                exit()
            sensor_name = line[1]
            current_windows = windowed_data.get(sensor_name)
            new_windows = PreprocessUtil.getWindowsFromLine(line)
            current_windows.extend(new_windows)
            windowed_data.update({sensor_name : current_windows })
        bar.next()
print()

#Dictionary ist voll. jetzt müssen die windows nur noch genullt werden...
#print("Differenziere Windows... ")
#for sensor in sensors:
#    windows = windowed_data.get(sensor)
#    nulled_windows = PreprocessUtil.differentiateWindows(windows, sensor)
#    windowed_data.update({ sensor: nulled_windows})

print("Normalisiere Windows...")
for sensor in sensors:
    windows = windowed_data.get(sensor)
    normalized_windows = PreprocessUtil.normalizeWindows(windows)
    windowed_data.update({ sensor: normalized_windows})
print()


#Kombiniere Alle Sensordaten in Kombo-Klasse
print("Kombiniere windows...")
combined_windows, kombo_name = PreprocessUtil.getCombinedWindowsOf32(windowed_data)
windowed_data.update({"Kombo" : combined_windows})
sensors.append("Kombo")
print("")


#Daten in die CSVs schreiben.
print("Schreibe CSVs...")
for sensor in sensors:
    writing_windows = windowed_data.get(sensor)
    PreprocessUtil.writeWindowsToFile(windowed_data.get(sensor), OUTPUTPATH + sensor + ".csv")
