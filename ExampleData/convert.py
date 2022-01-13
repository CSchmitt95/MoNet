import csv
import os
import math

from numpy import double

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
    for i in range(0, SAMPLES_PER_SECOND*WINDOW_SIZE_IN_S):
        for j in range(0, sensor_count):
            header.append("S" + str(j) + "_x" + str(i))
            header.append("S" + str(j) + "_y" + str(i))
            header.append("S" + str(j) + "_z" + str(i))
            header.append("S" + str(j) + "_w" + str(i))
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
"""

"""
Was ist mein Input?
    Eine Datei pro Bewegung mit Sensordaten von allen Sensoren.

Was soll mein Output sein nach der Vorverarbeitung?
    3 Dateien mit Trainingsdaten
        Sensor A 
            Eine Datei mit Trainingsdaten (Window)
        Sensor B
            Eine Datei mit Trainigsdaten (Window)
        SensorAB
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
        Wenn letzte Zeile:
            Windows extrahieren aber nur für validierung benutzen.       
'''


SAMPLES_PER_SECOND = 125
WINDOW_SIZE_IN_S = 3
QUATERNIONS_PER_SAMPLE = 4
SAMPLES_PER_WINDOW = SAMPLES_PER_SECOND*WINDOW_SIZE_IN_S

SENSOR_A_NAME = "Hand"
SENSOR_B_NAME = "Hosentasche"


data = readFile()

SensorA_Windows = []
SensorB_Windows = []
SensorAB_Windows = []

SensorA_TestWindows = []
SensorB_TestWindows = []
SensorAB_TestWindows = []

SensorA_Windows.append(generateWindowHeader(1))
SensorB_Windows.append(generateWindowHeader(1))
SensorAB_Windows.append(generateWindowHeader(2))

SensorA_TestWindows.append(generateWindowHeader(1))
SensorB_TestWindows.append(generateWindowHeader(1))
SensorAB_TestWindows.append(generateWindowHeader(2))

for line in data:
    movemnt_name = line[0]
    sensor_name = line[1]
    recording_id = line[2]
    
    if sensor_name == SENSOR_A_NAME:
        for searchline in data:
            if searchline[0] == movemnt_name and searchline[1] == SENSOR_B_NAME and searchline[2] == recording_id:
                if data.index(line) + 2 < len(data):
                    if data[data.index(line) + 2][0] != movemnt_name:
                        print("--> Test-Datenpaar gefunden! Movement: " + movemnt_name + " rec_id: " + recording_id) 
                        a_test_windows = getWindowsFromLine(line)
                        #print(str(len(a_windows)) + " Windows für Sensor A extrahiert")
                        b_test_windows = getWindowsFromLine(searchline)
                        #print(str(len(b_windows)) + " Windows für Sensor B extrahiert")
                        kombo_test_windows = combineWindows(a_test_windows, b_test_windows)
                        #print(str(len(kombo_test_windows)) +" Windows für Sensoren A,B kombiniert")

                        SensorA_TestWindows.extend(a_test_windows)
                        SensorB_TestWindows.extend(b_test_windows)
                        SensorAB_TestWindows.extend(kombo_test_windows)
                    else:
                        print("--> Datenpaar gefunden! Movement: " + movemnt_name + " rec_id: " + recording_id) 
                        a_windows = getWindowsFromLine(line)
                        #print(str(len(a_windows)) + " Windows für Sensor A extrahiert")
                        b_windows = getWindowsFromLine(searchline)
                        #print(str(len(b_windows)) + " Windows für Sensor B extrahiert")
                        kombo_windows = combineWindows(a_windows, b_windows)
                        #print(str(len(kombo_windows)) +" Windows für Sensoren A,B kombiniert")

                        SensorA_Windows.extend(a_windows)
                        SensorB_Windows.extend(b_windows)
                        SensorAB_Windows.extend(kombo_windows)
                        #print("Daten angehängt")
                        #print("Insgesamt Windows für  A: " + str(len(SensorA_Windows)))
                        #print("Insgesamt Windows für  B: " + str(len(SensorB_Windows)))
                        #print("Insgesamt Windows für AB: " + str(len(SensorAB_Windows)))
                else:
                    print("--> Test-Datenpaar gefunden! Movement: " + movemnt_name + " rec_id: " + recording_id) 
                    a_test_windows = getWindowsFromLine(line)
                    #print(str(len(a_windows)) + " Windows für Sensor A extrahiert")
                    b_test_windows = getWindowsFromLine(searchline)
                    #print(str(len(b_windows)) + " Windows für Sensor B extrahiert")
                    kombo_test_windows = combineWindows(a_test_windows, b_test_windows)
                    #print(str(len(kombo_test_windows)) +" Windows für Sensoren A,B kombiniert")

                    SensorA_TestWindows.extend(a_test_windows)
                    SensorB_TestWindows.extend(b_test_windows)
                    SensorAB_TestWindows.extend(kombo_test_windows)
                



print("Länge  A: " + str(len(SensorA_Windows)))
print("Länge  A Test: " + str(len(SensorA_TestWindows)) + " Ratio: " + (str(100*(len(SensorA_TestWindows)/len(SensorA_Windows)))))

print("Länge  B: " + str(len(SensorB_Windows)))
print("Länge  B Test: " + str(len(SensorB_TestWindows)) + " Ratio: " + (str(100*(len(SensorB_TestWindows)/len(SensorB_Windows)))))

print("Länge  Kombo: " + str(len(SensorAB_Windows)))
print("Länge  Kombo Test: " + str(len(SensorAB_TestWindows)) + " Ratio: " + (str(100*(len(SensorAB_TestWindows)/len(SensorAB_Windows)))))


SensorA_Windows = differentiateWindows(SensorA_Windows)
print("Schreibe Sensor A trainingsdaten...")
writeWindowsToFile(SensorA_Windows, "SensorA.csv")
SensorA_TestWindows = differentiateWindows(SensorA_TestWindows)
print("Schreibe Sensor A Testdaten...")
writeWindowsToFile(SensorA_TestWindows, "SensorA_Test.csv")

SensorB_Windows = differentiateWindows(SensorB_Windows)
print("Schreibe Sensor B trainingsdaten...")
writeWindowsToFile(SensorB_Windows, "SensorB.csv")
SensorB_TestWindows = differentiateWindows(SensorB_TestWindows)
print("Schreibe Sensor B Testdaten...")
writeWindowsToFile(SensorB_TestWindows, "SensorB_Test.csv")

SensorAB_Windows = differentiateWindows(SensorAB_Windows)
print("Schreibe Sensor AB trainingsdaten...")
writeWindowsToFile(SensorAB_Windows, "SensorsAB.csv")
SensorAB_TestWindows = differentiateWindows(SensorAB_TestWindows)
print("Schreibe Sensor AB Testdaten...")
writeWindowsToFile(SensorAB_TestWindows, "SensorAB_Test.csv")
