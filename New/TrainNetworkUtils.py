import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def saveHistoryGraph(history, filename):
    x = len(history.history['accuracy'])+1

    plt.plot(
        np.arange(1, x), 
        history.history['loss'], label='Loss'
    )
    plt.plot(
        np.arange(1, x), 
        history.history['accuracy'], label='Accuracy'
    )
    plt.plot(
        np.arange(1, x), 
        history.history['precision'], label='Precision'
    )
    plt.plot(
        np.arange(1, x), 
        history.history['recall'], label='Recall'
    )
    plt.title('Evaluation metrics', size=20)
    plt.xlabel('Epoch', size=14)
    plt.ylim(0, 1.5)
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf() 


def saveConfusionMatrix(cm_df, filename):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
