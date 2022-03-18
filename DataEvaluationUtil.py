from numpy import diagonal
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def balanceMatrix(df):
    df.reset_index()
    df.div(df.sum(axis=1)*100, axis=0)
    return df

def saveConfusionMatrix(name, cm_df, filename):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True, fmt='d')
    plt.title(name)
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    accuracy_text = "Accuracy: " + "{:4.2f}".format(getAccuracy(cm_df)) + "%"
    stolpern_text = "Stolpern\n Sens: " + str(getSensitivity(cm_df, 0)) + "% \n Spez: " + "{:4.2f}".format(getSpecificity(cm_df,0)) + "%\n"
    gehen_text = "Gehen\n Sens: " + str(getSensitivity(cm_df, 1)) + "% \n Spez: " + "{:4.2f}".format(getSpecificity(cm_df,1)) + "%\n"
    stehen_text = "Stehen\n Sens: " + str(getSensitivity(cm_df, 2)) + "% \n Spez: " + "{:4.2f}".format(getSpecificity(cm_df,2)) + "%\n"
    plt.figtext(0.35,-0.05,accuracy_text)
    plt.figtext(0.1,-0.25,stolpern_text)
    plt.figtext(0.4,-0.25,gehen_text)
    plt.figtext(0.7,-0.25,stehen_text)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()

def beatifyString(string):
    string = string.replace("_", " ")
    string = string.replace("grtl", "Gürtel")
    string = string.replace("hndglnk", "Handgelenk")
    string = string.replace("left", "Links")
    string = string.replace("right", "Rechts")    
    string = string.replace("raw", "raw ")     
    string = string.replace("raw", "Roh")    
    string = string.replace("nulled", "Genullt")
    string = string.replace("diff", "diff ")
    string = string.replace("diff", "Differenziert")

    return string

def getSensitivity(df, index):
    data = df.to_numpy()
    return data[index,index]

def getSpecificity(df, index):
    positives = df.sum(axis=0).to_numpy()[index]
    data = df.to_numpy()
    true_positives = data[index, index]
    ret = true_positives*100/positives
    if(positives == 0): 
        return 0
    return ret

def getAccuracy(df):
    whole = df.to_numpy()
    diagonal_sum = whole.trace()
    sum_x = whole.sum(axis = 1)
    sum = sum_x.sum()
    ret = diagonal_sum*100 / sum
    return ret