import matplotlib.pyplot as plt
import pandas as pd
import TrainNetworkUtils as trainUtils
import DataEvaluationUtil as dataUtils
import os
import numpy as np

index=True
outputdir="EvaluationResults/csv/"

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

#Diff Null Confidence 99%:
diffNulled_left_99_grtl = [[0,0,0,20],[0,0,0,120],[0,0,100,1]]
diffNulled_left_99_hndglnk = [[20,0,0,0],[4,4,0,112],[0,0,94,7]]
diffNulled_left_99_grtlhndglnk = [[18,0,0,2],[1,0,0,119],[0,0,98,3]]


#Diff Null Confidence 95%:
diffNulled_left_95_grtl = [[2,12,0,6],[11,87,4,18],[0,0,100,0]]
diffNulled_left_95_hndglnk = [[17,3,0,0],[11,79,0,30],[0,0,99,1]]
diffNulled_left_95_grtlhndglnk = [[16,4,0,0],[5,115,0,0],[0,0,100,0]]


#Diff Null Confidence 90%:
diffNulled_left_90_grtl = [[0,18,2,0],[12,99,9,0],[0,0,100,0]]
diffNulled_left_90_hndglnk = [[19,1,0,0],[1,105,0,14],[0,0,100,0]]
diffNulled_left_90_grtlhndglnk = [[15,5,0,0],[0,119,1,0],[0,0,100,0]]


df = pd.DataFrame(diffNulled_right_grtl, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "diffNulled_right_grtl.csv", index=index)
df = pd.DataFrame(diffNulled_right_hndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "diffNulled_right_hndglnk.csv", index=index)
df = pd.DataFrame(diffNulled_right_grtlhndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "diffNulled_right_grtlhndglnk.csv", index=index)
df = pd.DataFrame(diffNulled_left_grtl, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "diffNulled_left_grtl.csv", index=index)
df = pd.DataFrame(diffNulled_left_hndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "diffNulled_left_hndglnk.csv", index=index)
df = pd.DataFrame(diffNulled_left_grtlhndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "diffNulled_left_grtlhndglnk.csv", index=index)

df = pd.DataFrame(rawNulled_right_grtl, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "rawNulled_right_grtl.csv", index=index)
df = pd.DataFrame(rawNulled_right_hndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "rawNulled_right_hndglnk.csv", index=index)
df = pd.DataFrame(rawNulled_right_grtlhndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "rawNulled_right_grtlhndglnk.csv", index=index)
df = pd.DataFrame(rawNulled_left_grtl, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "rawNulled_left_grtl.csv", index=index)
df = pd.DataFrame(rawNulled_left_hndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "rawNulled_left_hndglnk.csv", index=index)
df = pd.DataFrame(rawNulled_left_grtlhndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "rawNulled_left_grtlhndglnk.csv", index=index)

df = pd.DataFrame(raw_right_grtl, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "raw_right_grtl.csv", index=index)
df = pd.DataFrame(raw_right_hndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "raw_right_hndglnk.csv", index=index)
df = pd.DataFrame(raw_right_grtlhndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "raw_right_grtlhndglnk.csv", index=index)
df = pd.DataFrame(raw_left_grtl, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "raw_left_grtl.csv", index=index)
df = pd.DataFrame(raw_left_hndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "raw_left_hndglnk.csv", index=index)
df = pd.DataFrame(raw_left_grtlhndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen"})
df.to_csv(outputdir + "raw_left_grtlhndglnk.csv", index=index)

df = pd.DataFrame(diffNulled_left_99_grtl, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen","DontKnow"})
df.to_csv(outputdir + "diffNulled_left_99_grtl.csv", index=index)
df = pd.DataFrame(diffNulled_left_99_hndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen","DontKnow"})
df.to_csv(outputdir + "diffNulled_left_99_hndglnk.csv", index=index)
df = pd.DataFrame(diffNulled_left_99_grtlhndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen","DontKnow"})
df.to_csv(outputdir + "diffNulled_left_99_grtlhndglnk.csv", index=index)

df = pd.DataFrame(diffNulled_left_95_grtl, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen","DontKnow"})
df.to_csv(outputdir + "diffNulled_left_95_grtl.csv", index=index)
df = pd.DataFrame(diffNulled_left_95_hndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen","DontKnow"})
df.to_csv(outputdir + "diffNulled_left_95_hndglnk.csv", index=index)
df = pd.DataFrame(diffNulled_left_95_grtlhndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen","DontKnow"})
df.to_csv(outputdir + "diffNulled_left_95_grtlhndglnk.csv", index=index)

df = pd.DataFrame(diffNulled_left_90_grtl, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen","DontKnow"})
df.to_csv(outputdir + "diffNulled_left_90_grtl.csv", index=index)
df = pd.DataFrame(diffNulled_left_90_hndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen","DontKnow"})
df.to_csv(outputdir + "diffNulled_left_90_hndglnk.csv", index=index)
df = pd.DataFrame(diffNulled_left_90_grtlhndglnk, {"Stolpern","Gehen","Stehen"},{"Stolpern","Gehen","Stehen","DontKnow"})
df.to_csv(outputdir + "diffNulled_left_90_grtlhndglnk.csv", index=index)
