import csv
import os
import math

from numpy import double

SAMPLES_PER_SECOND = 125
WINDOW_SIZE_IN_S = 3
QUATERNIONS_PER_SAMPLE = 4
SAMPLES_PER_WINDOW = SAMPLES_PER_SECOND*WINDOW_SIZE_IN_S

SENSOR_A_NAME = "Hand"
SENSOR_B_NAME = "Hosentasche"


def readFile():
    directory = 'input/'
    return_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            print("datei " + filename + " gefunden...")
            file = os.path.join(directory, filename)
            with open(file, newline='') as input:
                reader = csv.reader(input)
                data = list(reader)
                return_data.extend(data[1:])
    return return_data
            

#Macht aus eingelesenen Daten Fenster für Ein-Sensor-Training
def TurnDataIntoWindows(data):
    windows = []
    for line in data:
        
        name = line[0]
        pure_data = line[3:]
        i = 0
        while i < (len(pure_data) - SAMPLES_PER_WINDOW):
            new = []
            new.append(name)
            new.extend(pure_data[i:i+SAMPLES_PER_WINDOW])
            i = i+1
            windows.append(new)
            #print("Fenstergröße ist: " + str(len(new)))

        print("Aus Bewegung " + name + " " + str(i) + " Fenster erstellt")
    
    print("Insgesamt " + str(len(windows)) + " Fenster generiert")
    return windows

#Holt Daten eines des angegebenen Sensors 
def getDataForSensor(data, SensorName):
    returnData = []
    for line in data:
        if line[1] == SensorName:
            returnData.append(line)
    return returnData

#Schreibt Fenster in eine Datei. 
def writeWindowsToFile(windows, filename):
        with open("output/" + filename, "w", newline="") as fileoutput:
            writer = csv.writer(fileoutput)
            writer.writerows(windows)

#Prüft ob die Fenster in valider Fensterform sind
def checkIfValid(windows):
    returnval = True
    for window in windows:
        if len(window) != 601:
            returnval = False
    return returnval

#Nimmt zwei Windows auf und gibt kombiniertes Traingsdatenset zurück
def getDataForSensors(data, sensor1, sensor2):
    returnData = []
    for line in data:
        if line[1] == sensor1:
            for searchline in data:
                if line[2] == searchline[2] and searchline[1] == sensor2:
                    print("bla")


def getWindowsFromLine(line):
    windows = []
    name = line[0]
    pure_data = line[3:]
    i = 0
    while i < (len(pure_data) - SAMPLES_PER_WINDOW*QUATERNIONS_PER_SAMPLE):
        new = []
        new.append(name)
        new.extend(pure_data[i:i+SAMPLES_PER_WINDOW*QUATERNIONS_PER_SAMPLE])
        i = i+QUATERNIONS_PER_SAMPLE
        windows.append(new)
        #print("Fenstergröße ist: " + str(len(new)))
    return windows

def combineWindows(a_windows, b_windows):
    #print("-----> A: " + str(len(a_windows)) + " B: " + str(len(b_windows)))
    windows = []
    i = 0
    for i in range(0, len(a_windows)):
        #print("->A: " + str(len(a_windows[i])) + " B: " + str(len(b_windows[i])))    
        kombo_window = []
        kombo_window.extend(a_windows[i])
        #print(kombo_window)
        kombo_window.extend(b_windows[i][1:])
        #print(len(kombo_window))
        windows.append(kombo_window)
        i = i+1
    return windows

def generateWindowHeader(sensor_count):
    header = []
    header.append("MovementName")
    for i in range(0, sensor_count):
        for j in range(0, SAMPLES_PER_SECOND*WINDOW_SIZE_IN_S):
            header.append("S" + str(i) + "_x" + str(j))
            header.append("S" + str(i) + "_y" + str(j))
            header.append("S" + str(i) + "_z" + str(j))
            header.append("S" + str(i) + "_w" + str(j))
    return header

def differentiateWindows(windows):
    for window in windows[1:]:
        for i in range(len(window)-4, 0, -4):
            if i == 1:
                window[1] = 0
                window[2] = 0
                window[3] = 0
                window[4] = 0
            else:
                window[i+0] = float(window[i+0]) - float(window[i-4+0])
                window[i+1] = float(window[i+1]) - float(window[i-4+1])
                window[i+2] = float(window[i+2]) - float(window[i-4+2])
                window[i+3] = float(window[i+3]) - float(window[i-4+3])
    return windows


"""
Data:           MovementName, SensorName, Recording_id, [Sensordaten] 
Window:         Movementname, 2400x[Sensordaten]
DualWindow:     Movementname, 2400x[Sensordaten_1], 2400x[Sensordaten_2]
"""

"""
Was ist mein Input?
    Eine Datei pro Bewegung mit Sensordaten von allen Sensoren.

Was soll mein Output sein nach der Vorverarbeitung?
    3 Ordner mit Trainingsdaten
        Handgelenk
            Eine Datei mit Trainingsdaten (Window)
        Gürtel
            Eine Datei mit Trainigsdaten (Window)
        Kombiniert
            Eine Datei mit Trainingsdaten (DualWindow)
"""


'''
Pseudo Algorithmus

    Wir wollen drei Sets voll bekommen A, B und Kombo
    Wir gehen in die Datei Rein
    In jeder Zeile
        Namen der Bewegung erfassen
        Wenn die Daten von Sensor A kommen:
            Gucken ob Sensor B das gegenstück hat.
                Windows für Sensor A holen.
                Windows für Sensor B holen.
                aus den beiden die Kombo Windows 
        
'''

data = readFile()

SensorA_Windows = []
SensorB_Windows = []
SensorAB_Windows = []

SensorA_Windows.append(generateWindowHeader(1))
SensorB_Windows.append(generateWindowHeader(1))
SensorAB_Windows.append(generateWindowHeader(2))

for line in data:
    movemnt_name = line[0]
    sensor_name = line[1]
    recording_id = line[2]
    
    if sensor_name == SENSOR_A_NAME:
        for searchline in data:
            if searchline[0] == movemnt_name and searchline[1] == SENSOR_B_NAME and searchline[2] == recording_id:
                print("--> Datenpaar gefunden! Movement: " + movemnt_name + " rec_id: " + recording_id) 
                a_windows = getWindowsFromLine(line)
                print(str(len(a_windows)) + " Windows für Sensor A extrahiert")
                b_windows = getWindowsFromLine(searchline)
                print(str(len(b_windows)) + " Windows für Sensor B extrahiert")
                kombo_windows = combineWindows(a_windows, b_windows)
                print(str(len(kombo_windows)) +" Windows für Sensoren A,B kombiniert")

                SensorA_Windows.extend(a_windows)
                SensorB_Windows.extend(b_windows)
                SensorAB_Windows.extend(kombo_windows)
                print("Daten angehängt")
                print("Insgesamt Windows für  A: " + str(len(SensorA_Windows)))
                print("Insgesamt Windows für  B: " + str(len(SensorB_Windows)))
                print("Insgesamt Windows für AB: " + str(len(SensorAB_Windows)))

"""
print("Länge  A: " + str(len(SensorA_Data)))
print("Länge  B: " + str(len(SensorB_Data)))
print("Länge  Kombo: " + str(len(Kombo_Data)))
"""
#SensorA_Windows = differentiateWindows(SensorA_Windows)
#print("Schreibe Sensor A trainingsdaten...")
#writeWindowsToFile(SensorA_Windows, "SensorA.csv")
SensorB_Windows = differentiateWindows(SensorB_Windows)
print("Schreibe Sensor B trainingsdaten...")
writeWindowsToFile(SensorB_Windows, "SensorB.csv")
SensorAB_Windows = differentiateWindows(SensorAB_Windows)
print("Schreibe Sensor AB trainingsdaten...")
writeWindowsToFile(SensorAB_Windows, "SensorsAB.csv")

"""
"""