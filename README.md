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
Aber da für die Präsentation der Ergebnisse die Reihenfolge der Versuche verändert wurde und auch manche Ergebnisse aufgeteilt wurden, darf man nicht von der Versuchsnummer aus der Arbeit auf den Namen in diesem Repository schließen.

Eine kurze Übersicht: 
- Versuch 1 Laufzeit und Versuch 5 Echtzeit-Laufzeit beziehen ihre daten aus 0V2
- Versuch 2 Speicherauslastung bezieht seine Daten aus 0V1
- Versuch 3 Vorverarbeitung bezieht seine daten aus 0V5
- Versuch 4 Sensorposition bezieht seine daten aus 0V6
- Versuch 5 Echtzeit Evaluation bezieht seine Daten auch aus 0V6

Auch sind einige der Datensätze nicht miteinander kompatibel, da sich die Bezeichnungen der Sensoren bezüglich der Benutzung von Umlauten verändert haben.

In diesem Repository werden die Trainingsdaten nicht getrackt, da sie zu viel Platz wegnehmen.
Um sie wieder herzustellen muss das Preprocess.py Skript ausgeführt werden.
Je nach suffix müssen dafür gewissen Teile des Skripts auskommentiert werden.

- "DiffNull"  - Kein Auskommentieren nötig
- "NoDiffNull" - Zeilen 80-81 müssen auskommentiert werden.
- "DiffNoNull" - Zeilen 108-113 müssen auskommentiert werden.
- "NoDiffNoNull" - Zeilen 80-81 und 108-113 müssen auskommentiert werden.


Um die Skripte auszuführen wird python3 mit den Bibliotheken aus der requirements.txt benötigt.
Die Bibliotheken können mit folgendem Befehl installiert werden:
python3 -m pip install -r .\reqs.txt