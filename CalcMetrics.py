from audioop import avg
import numpy as np  
import pandas as pd


MATRIX = input("Welches Ergebnis soll gecheckt werden?\n")
df = pd.read_csv(MATRIX)
df = df.iloc[: , 1:]

labels = list(df.columns.values)

print("Labels: ")
print(labels)
diagsum = 0
avg_recall = 0 
avg_precision = 0

for index, row in df.iterrows():
    #print(index)
    #print(" ")
    #print(row)

    print("Label: "+  labels[index])
    diagonal = row.iloc[index]
    row_sum = row.sum();


    diagsum = row.iloc[index] + diagsum
    #print("Summe: " + str(sum))
    recall = diagonal / row_sum
    precision = diagonal / df.sum()[index]

    avg_recall = avg_recall + recall
    avg_precision = avg_precision + precision

    print("---->Recall: " + str(recall))
    print("---->Precision: " + str(precision))
    print("")


accuracy = diagsum / df.to_numpy().sum()
avg_recall = avg_recall/len(df.index)
avg_precision = avg_precision/len(df.index)


print("Accuracy: " + str(accuracy))
print("Avg Recall: " + str(avg_recall))
print("Avg Precision: " + str(avg_precision))