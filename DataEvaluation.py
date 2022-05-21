from operator import index
import matplotlib.pyplot as plt
import pandas as pd
import TrainModelUtils as trainUtils
import DataEvaluationUtil as dataUtils
import os
import numpy as np

#input_dir = "EvaluationResults/csv/"
#output_dir = "EvaluationResults/png/"

ROOT = "EvaluationResults/"
INPUT_FOLDER = input("Welche Daten sollen visualisiert werden?")

input_dir = INPUT_FOLDER #ROOT + INPUT_FOLDER
output_dir = input_dir+"/"

filecounter = 0
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        filecounter = filecounter+1
        file = os.path.join(input_dir, filename)
        df = pd.read_csv(file, index_col=0)
        print(df)
        df = df.div(df.sum(axis=1), axis=0)
        df = df.mul(100, axis=0)
        df = df.round(0)
        df = df.astype(np.int32)
        figname = dataUtils.beatifyString(filename[:-4])
        dataUtils.saveConfusionMatrix(figname,df,output_dir+filename[:-4]+".png")
        