# Team 404 Name not found
Teammitglieder: Ellen, Jia und Noah

## Model 4: XGBoost Classifier + ResNet 
Model wird als four_classes classifier trainiert. Wichtig fürs Training sind die 
1. features.json Datei
2. utils.py  
3. config
4. die Ordner Keras_models und models
da sie im gleichen Ordner liegen muss, wie die train.py und predict.py. 
Im Keras_models Ordner befindet sich das trainierte ResNet

Für den vollständigen Download der Inhalte des trainierten ResNet50 im Keras_models-Ornder muss das Projekt mit Git Bash
heruntergeladen werden. 

In der LICENSE-Datei ist das Copyright für das ResNet hinterlegt

# TensorFlow2.0_ResNet

See https://github.com/calmisential/Basic_CNNs_TensorFlow2.0 for more CNNs.
+ [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152](https://github.com/calmisential/TensorFlow2.0_ResNet)
+ [InceptionV3](https://github.com/calmisential/TensorFlow2.0_InceptionV3)


## References
1. The original paper: https://arxiv.org/abs/1512.03385
2. The TensorFlow official tutorials: https://tensorflow.google.cn/beta/tutorials/quickstart/advanced

# XGB-Classifier
## hrv-analysis
Features wurden aus verschiedenen Domänen extrahiert:
- Time domain
- Frequency domain
- Geometrical Features
- Poincare-plot Features
- Csi-csv Features
- Sample entropy

Source: https://aura-healthcare.github.io/hrv-analysis/hrvanalysis.html#module-hrvanalysis.extract_features
## tsfel
Für Spectral Features:
- Fundamental frequency
- Human range energy
- Max power spectrum, Maximum Frequency, Median frequency
- Power bandwidth
- Spectral centroid, Spectral decrease, Spectral entropy, Spectral kurtosis, Spectral positive turning points, 
Spectral roll-off, Spectral roll-on, Spectral skewness, Spectral spread, Spectral variation
- Wavelet absolute mean

Source: https://github.com/fraunhoferportugal/tsfel
## Hyperparametertuning
Die Hyperparameter wurden nach folgendem Guide durch eine GridSearch gewählt: 
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
