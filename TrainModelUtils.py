import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

def saveHistoryGraph(history, filename):
    x = len(history.history['accuracy'])+1
    plt.figure(figsize=(18,8))
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
    plt.title('Trainingsverlauf', size=20)
    plt.xlabel('Epoche', size=14)
    plt.ylim(0, 1.5)
    plt.legend()
    plt.savefig(filename + "_history.png", bbox_inches='tight')
    plt.clf() 
    with open(filename + "_stats.txt", "w") as text_file:
        text_file.write("Final Stats:\n")
        text_file.write("Loss: " + str(history.history['loss'][len(history.history['loss'])-1]) + "\n")
        text_file.write("Accuracy: " + str(history.history['accuracy'][len(history.history['accuracy'])-1]) + "\n")
        text_file.write("Precision: " + str(history.history['precision'][len(history.history['precision'])-1]) + "\n")
        text_file.write("Recall: " + str(history.history['recall'][len(history.history['recall'])-1]) + "\n")


def saveConfusionMatrix(cm_df, filename):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()

def saveDistributionGraph(df, filename):
    df['MovementName'].value_counts().plot(kind='bar')
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()


def writeReport(lerning_rate, number_of_epochs, batch_size, model, filename, movements):
    with open(filename, "w") as text_file:
        text_file.write("Movements: " + str(movements) + "\n")
        text_file.write("Learning Rate: %s\n" % lerning_rate)
        text_file.write("Number of Epochs: %s\n" % number_of_epochs)
        text_file.write("Batch_size: %s\n" % batch_size)
        text_file.write("\n")
        text_file.write("Model Summary: \n")
        text_file.write(get_model_summary(model))


def get_model_summary(model: tf.keras.Model) -> str:
    string_list = []
    model.summary(line_length=80, print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list)


def saveDistributionGraph32(df, movements, filename):
    df[movements].sum().plot(kind='bar')
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
