# Team 404 Name not found
Teammitglieder: Ellen, Jia und Noah

## Model 2: XGBoost Classifier
Model wird als four_classes classifier trainiert. Wichtig fürs Training ist die features.json Datei, da sie im gleichen 
Ordner liegen muss, wie die train.py und predict.py - wird für die Feature Extraction über die tsfel-library benötigt. 


## Wichtig!

Bitte achtet bei der Abgabe darauf, dass alle von uns gestellten Dateien auf dem Top-Level des Repositories liegen. Testet die Funktionsfähigkeit eures Codes mit dem Skript predict_pretrained.py. 

Die Dateien 
- predict_pretrained.py
- wettbewerb.py
- score.py

werden von uns beim testen auf den ursprünglichen Stand zurückgesetzt. Es ist deshalb nicht empfehlenswert diese zu verändern. In predict.py ist für die Funktion `predict_labels` das Interface festgelegt, das wir für die Evaluierung verwenden.

`predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=False) -> List[Tuple[str,str]]`

Insbesondere den `model_name` könnt ihr verwenden um bei der Abgabe verschiedene Modelle zu kennzeichnen, welche zum Beispiel durch eure Ordnerstruktur dargestellt werden. Der Parameter `is_binary_classifier` ermöglicht es zu entscheiden, ob mit dem Modell nur die zwei Hauptlabels "Atrial Fibrillation ['A']" und "Normal ['N']" klassfiziert werden (binärer Klassifikator), oder alle vier Label.

Bitte gebt alle verwendeten packages in "requirements.txt" bei der Abgabe zur Evaluation an und testet dies vorher in einer frischen Umgebung mit `pip install -r requirements.txt`. Als Basis habt ihr immer die vorgegebene "requirements.txt"-Datei. Wir selbst verwenden Python 3.8. Wenn es ein Paket gibt, welches nur unter einer anderen Version funktioniert ist das auch in Ordung. In dem Fall bitte Python-Version mit angeben.
