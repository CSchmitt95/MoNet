import matplotlib.pyplot as plt
import pandas as pd
import TrainNetworkUtils as trainUtils
import DataEvaluationUtil as dataUtils
import os
import numpy as np


#Raw 
raw_right_grtl = [[6,1,3],[0,120,0],[47,55,0]]
raw_right_hndglnk = [[10,0,0],[120,0,0],[102,0,0]]
raw_right_grtlhndglnk = [[9,1,0],[0,120,0],[51,51,0]]


raw_left_grtl = [[10,0,0],[0,120,0],[6,54,0]]
raw_left_hndglnk = [[10,0,0],[120,0,0],[60,0,0]]
raw_left_grtlhndglnk = [[4,6,0],[0,120,0],[0,60,0]]

#Raw Nulled
rawNulled_right_grtl = [[2,5,3],[17,89,14],[11,7, 82]]
rawNulled_right_hndglnk = [[9,1,0],[6,114,0],[20,0,80]]
rawNulled_right_grtlhndglnk = [[1,9,0],[4,116,0],[23,9,68]]
 

rawNulled_left_grtl = [[4,6,1 ],[7,107,6],[13,9,78]]
rawNulled_left_hndglnk = [[1,1,9],[29,57,34],[8,0,92]]
rawNulled_left_grtlhndglnk = [[3,0,8],[20,34,66],[11,6,83]]

#Diff Nulled
diffNulled_right_grtl = [[0, 9, 1],[0,122,3],[0,0,100]]
diffNulled_right_hndglnk = [[10,0,0],[47,78,0],[0,0,100]]
diffNulled_right_grtlhndglnk = [[8,2,0],[0,123,2],[0,0,100]]

diffNulled_left_grtl = [[2,6,2],[0,116,4],[0,0,100]]
diffNulled_left_hndglnk = [[0,2,8],[2,116,4],[0,0,100]]
diffNulled_left_grtlhndglnk = [[0,8,2],[0,114,6],[0,0,100]]


df = pd.DataFrame(diffNulled_right_grtl, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/diffNulled_right_grtl.csv", index=False)
df = pd.DataFrame(diffNulled_right_hndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/diffNulled_right_hndglnk.csv", index=False)
df = pd.DataFrame(diffNulled_right_grtlhndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/diffNulled_right_grtlhndglnk.csv", index=False)
df = pd.DataFrame(diffNulled_left_grtl, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/diffNulled_left_grtl.csv", index=False)
df = pd.DataFrame(diffNulled_left_hndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/diffNulled_left_hndglnk.csv", index=False)
df = pd.DataFrame(diffNulled_left_grtlhndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/diffNulled_left_grtlhndglnk.csv", index=False)

df = pd.DataFrame(rawNulled_right_grtl, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/rawNulled_right_grtl.csv", index=False)
df = pd.DataFrame(rawNulled_right_hndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/rawNulled_right_hndglnk.csv", index=False)
df = pd.DataFrame(rawNulled_right_grtlhndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/rawNulled_right_grtlhndglnk.csv", index=False)
df = pd.DataFrame(rawNulled_left_grtl, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/rawNulled_left_grtl.csv", index=False)
df = pd.DataFrame(rawNulled_left_hndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/rawNulled_left_hndglnk.csv", index=False)
df = pd.DataFrame(rawNulled_left_grtlhndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/rawNulled_left_grtlhndglnk.csv", index=False)

df = pd.DataFrame(raw_right_grtl, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/raw_right_grtl.csv", index=False)
df = pd.DataFrame(raw_right_hndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/raw_right_hndglnk.csv", index=False)
df = pd.DataFrame(raw_right_grtlhndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/raw_right_grtlhndglnk.csv", index=False)
df = pd.DataFrame(raw_left_grtl, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/raw_left_grtl.csv", index=False)
df = pd.DataFrame(raw_left_hndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/raw_left_hndglnk.csv", index=False)
df = pd.DataFrame(raw_left_grtlhndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv("EvaluationResults/raw_left_grtlhndglnk.csv", index=False)



directory = "EvaluationResults/csv/"
filecounter = 0
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filecounter = filecounter+1
        file = os.path.join(directory, filename)
        df = pd.read_csv(file)
        df = df.div(df.sum(axis=1), axis=0)
        df = df.mul(100, axis=0)
        df = df.round(0)
        df = df.astype(np.int32)
        figname = dataUtils.beatifyString(filename[:-4])
        dataUtils.saveConfusionMatrix(figname,df,directory+filename[:-4]+".png")
        