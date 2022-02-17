from re import L
from symtable import Symbol

from numpy import isin
import numpy as np
import PreprocessConstants
import csv
import copy
from progress.bar import Bar
from pyquaternion import Quaternion
from numba import jit
import matplotlib.pyplot as plt


def getWindowsFromLine(line):
    windows = []
    name = line[0]
    pure_data = line[3:]
    i = 0
    while i < (len(pure_data) - PreprocessConstants.SAMPLES_PER_WINDOW*PreprocessConstants.QUATERNIONS_PER_SAMPLE):
        new = []
        new.append(name)
        new.extend(pure_data[i:i+PreprocessConstants.SAMPLES_PER_WINDOW*PreprocessConstants.QUATERNIONS_PER_SAMPLE])
        i = i+PreprocessConstants.QUATERNIONS_PER_SAMPLE
        windows.append(new)
        #print("Fenstergröße ist: " + str(len(new)))
    return windows

def getWindowsFromLine32(line, movements):
    name = line[0]
    pure_data = line[3:]

    shape_x = len(movements) +  PreprocessConstants.QUATERNIONS_PER_WINDOW
    shape_y = int((len(pure_data) - PreprocessConstants.QUATERNIONS_PER_WINDOW)/4)
    windows = np.empty(shape=(shape_y,shape_x))

    #Für Jede Zeile im neuen array
    for row in range(0, shape_y):
        #Für die Label spalten...
        for j in range(0, len(movements)):
            if movements[j] == name:
                windows[row,j] = 1
            else:        
                windows[row,j] = 0
        #Für die Daten Spalten...
        for k in range(0, PreprocessConstants.QUATERNIONS_PER_WINDOW):
            windows[row, len(movements) + k] = pure_data[row*4 + k]
    return windows
    
def getNumberOfWindowsFromLine32(line):
    pure_data = line[0:3]
    return int((len(pure_data) - PreprocessConstants.QUATERNIONS_PER_WINDOW)/4)

def generateWindowHeader(sensor_number):
    header = []
    if sensor_number == 0:
        header.append("MovementName")
    for i in range(0, PreprocessConstants.SAMPLES_PER_SECOND*PreprocessConstants.WINDOW_SIZE_IN_S):
        header.append("S%d_w%03d" % (sensor_number, i))
        header.append("S%d_x%03d" % (sensor_number, i))
        header.append("S%d_y%03d" % (sensor_number, i))
        header.append("S%d_z%03d" % (sensor_number, i))
    return header


@jit(nopython=True)
def differentiateWindows(windows, name):
    with Bar(name, max=len(windows)) as bar:
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
            bar.next()
        return windows

def differentiateCorrectly(lines):
    for line in lines:
        name = line[0]
        data = line[3:]
        with Bar(name + " (" + str(lines.index(line)) + "/" + str(len(lines))+ ")", max=int(len(data)/4)) as bar:
            for i in range(0, int(len(data)/4 -4)):
                this = Quaternion(data[i*4],data[i*4+1],data[i*4+2],data[i*4+3])
                next = Quaternion(data[(i+1)*4],data[(i+1)*4+1],data[(i+1)*4+2],data[(i+1)*4+3])
                diff = next * this.inverse
                data[i*4+0] = float(diff[0])
                data[i*4+1] = float(diff[1])
                data[i*4+2] = float(diff[2])
                data[i*4+3] = float(diff[3])
                bar.next()
        line[3:] = data
            
@jit(nopython=True)
def getCombinedWindowsOf(windowed_data):
    sensors = windowed_data.keys()
    output_windows = []
    first = True
    counter = 0
    for sensor in sensors:
        sensor_windows = copy.deepcopy(windowed_data.get(sensor))
        if first == True:
            print("Erster Sensor: " + sensor)
            output_windows = sensor_windows
            first = False
        else:
            print("Füge Hinzu: " + sensor)
            for i in range(0, len(output_windows)):
                if i == 0:
                    output_windows[i].extend(generateWindowHeader(counter))
                else:
                    output_windows[i].extend(sensor_windows[i][1:])
        counter = counter + 1

    return output_windows

#Schreibt Fenster in eine Datei. 
def writeWindowsToFile(windows, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as fileoutput:
        writer = csv.writer(fileoutput)
        with Bar(filename, max=len(windows)) as bar:
            for row in windows:
                output = np.array(row)
                writer.writerow(row)
                bar.next()



def ListAnalyser(object):
    print("Diese liste hat " + str(len(object)) + " Elemente")
    for x in object:
        if(isinstance(x, list)):
            print("--> Element ist Liste(" + str(len(x))+ ")")

def normalizeWindows(windows):    
    with Bar("normalisiere Windows", max=len(windows)) as bar:
        for window in windows[1:]:
            window_data = window
            for i in range(1, int((len(window_data)-1)/4)):
                window_data[i*4+1] = float(window_data[i*4+1]) - float(window_data[1])
                window_data[i*4+2] = float(window_data[i*4+2]) - float(window_data[2])
                window_data[i*4+3] = float(window_data[i*4+3]) - float(window_data[3])
                window_data[i*4+4] = float(window_data[i*4+4]) - float(window_data[4])
            
            window_data[1] = 0
            window_data[2] = 0
            window_data[3] = 0
            window_data[4] = 0
            bar.next()
            
    return windows

def normalizeWindows32(windows):
    movement_count = len(windows[0]) % PreprocessConstants.QUATERNIONS_PER_WINDOW
    with Bar("normalisiere Windows", max=len(windows)) as bar:
        for row in windows:
            for i in range(1, int((len(row)-1)/4)):
                row[i*4+0+movement_count] = row[i*4+0+movement_count] - row[0+movement_count]
                row[i*4+1+movement_count] = row[i*4+1+movement_count] - row[1+movement_count]
                row[i*4+2+movement_count] = row[i*4+2+movement_count] - row[2+movement_count]
                row[i*4+3+movement_count] = row[i*4+3+movement_count] - row[3+movement_count]
            
            row[0+movement_count] = 0
            row[1+movement_count] = 0
            row[2+movement_count] = 0
            row[3+movement_count] = 0
            bar.next()
            
    return windows

def generateWindowHeader32(movements, number_of_sensors):
    header = ""
    first = True
    for movement in movements:
        if first:
            header = movement
            first = False
        else:
            header = header + "," + movement
    for i in range (0, number_of_sensors):
        for j in range (0, PreprocessConstants.SAMPLES_PER_SECOND*PreprocessConstants.WINDOW_SIZE_IN_S):
            header = header + ",S%d_w%03d,S%d_x%03d,S%d_y%03d,S%d_z%03d" % (i, j, i, j, i, j, i, j)
    return header


def writeWindowsToFile32(windows, filename, movements):
    sensorcount = int(len(windows[0])/1500)
    np.savetxt(filename, windows, delimiter=",", fmt="%.7f", header=generateWindowHeader32(movements=movements, number_of_sensors=sensorcount))


def getCombinedWindowsOf32(windowed_data):
    sensors = windowed_data.keys()
    kombo_name = ""
    number_of_sensors = len(sensors)
    s = list(sensors)[0]
    sensor_data = windowed_data.get(s)

    number_of_movements = len(sensor_data[0]) % PreprocessConstants.QUATERNIONS_PER_WINDOW
    width = number_of_sensors*PreprocessConstants.QUATERNIONS_PER_WINDOW + number_of_movements
    height = len(sensor_data)
    #np.array([], dtype=np.float32).reshape(0, movementcount + PreprocessConstants.QUATERNIONS_PER_WINDOW)
    output_windows = np.array([], dtype=np.float32).reshape(height, 0)
    
    pos = 0
    first = True
    for sensor in sensors:
        kombo_name = kombo_name + sensor
        if first:
            data = windowed_data.get(sensor)
            first = False
        else:
            data = np.delete(windowed_data.get(sensor),np.s_[0:number_of_movements],axis=1)        
        
        output_windows = np.concatenate((output_windows,data), axis=1 )

    return output_windows, kombo_name