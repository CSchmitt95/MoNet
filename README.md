# MoNet

MoNet ist die Trainingsumgebung für TensorFlow Modelle auf Basis der MoRec App.

Nachdem man mit MoRec Daten aufgezeichnet und Exportiert hat, kann man in wenigen Schritten ein Modell aus den Daten trainieren.
1. kopiere den Ordner der Sitzung vom Android Telefon nach Data/ExportedData/
2. Starte die Vorverarbeitung mit Py -3.8 ./PreprocessData.py
3. Gib einen Suffix für die Vorverarbeitungsmethode an, falls diese Verändert wurde.
4. Starte das Training des Neuronalen Netzes Py -3.8 ./TrainModel.py
5. Gib den Namen für den Trainingsdatensatz an. Der Setzt sich aus [Sitzungsname]_[Preprocess-Suffix] zusammen.
6. Gib ein Suffix für das Training an, falls die Trainingsparameter verändert wurden.

Herzlichen Glückwunsch. Im Ordner TrainingResults findest du einen Ordner mit dem Namen [Sitzungsname]_[Preprocess-Suffix]_[Training-Suffix] die Trainingsergebnisse.
Im Unterordner Model findest du für jeden Sensor das Modell als TensorFlow und TensorFlow-Lite Modell.
Im Unterordner Graphs gibt es einen Graphen für die Trainingshistorie.

Das Trainierte Modell kann mit jedem vorverarbeiteten Datensatz evaluiert werden.
1. Starte Py -3.8 ./EvaluateModel.py
2. Gib nun den Namen des zu Evalierenden Modells ein
3. Gib nun den Namen des Vorverarbeiteten Datensatzes ein der Verwendet werden soll.

Im Ordner EvaluationResults/[Sitzungsname]_[Preprocess-Suffix]_[Training-Suffix]/[Sitzungsname_Testset]_[Preprocess-Suffix]/ liegen nun die Ergebnisse der Evaluation.
Für jedes Modell wird eine Konfusionsmatrix in einer CSV Datei erstellt.



Achtung:
Die Datensätze mit den Namen 0VX enthalten die Daten für die Finalen Versuche.
Aber da für die Präsentation der Ergebnisse die Reihenfolge der Versuche verändert wurde und auch manche Ergebnisse aufgeteilt wurden, darf man nicht von der Versuchnummer aus der Arbeit auf den Namen in diesem Repository schließen.
