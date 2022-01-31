from symtable import Symbol

from numpy import isin
import PreprocessConstants
import csv
from progress.bar import Bar

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

def generateWindowHeader(sensor_number):
    header = []
    if sensor_number == 0:
        header.append("MovementName")
    for i in range(0, PreprocessConstants.SAMPLES_PER_SECOND*PreprocessConstants.WINDOW_SIZE_IN_S):
        header.append("S%d_x%03d" % (sensor_number, i))
        header.append("S%d_y%03d" % (sensor_number, i))
        header.append("S%d_z%03d" % (sensor_number, i))
        header.append("S%d_w%03d" % (sensor_number, i))
    return header

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

def getCombinedWindowsOf(windowed_data):
    sensors = windowed_data.keys()
    output_windows = []
    first = True
    counter = 0
    for sensor in sensors:
        sensor_windows = windowed_data.get(sensor)
        if first == True:
            output_windows = sensor_windows
            output_windows[0] = generateWindowHeader(0)
            first = False
        else:
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
                writer.writerow(row)
                bar.next()


def ListAnalyser(object):
    print("Diese liste hat " + str(len(object)) + " Elemente")
    for x in object:
        if(isinstance(x, list)):
            print("--> Element ist Liste(" + str(len(x))+ ")")

