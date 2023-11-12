# X_AI_Hackathon

Repository for the hackathon team #6 "Code with the bros"

Les dossiers .streamlit et assets permettent de coder la webapp. 
Pour l'exécuter, exécuter la commande dans le terminal à l'emplacement du fichier accueil : streamlit run _Accueil.py

Le dossier pipeline met en oeuvre une pipeline pour créer un réseau de CNN entraîné sur le dataset fourni.
Dans ce dossier :
 - model.py: code le modèle
 - data.py: permet de load les data, de les split et les augmenter.
 - train.py: code les fonctions d'entraînement. Avec cross validation qui sélectionne plusieurs modèles et on s'appuit sur la moyenne de ces modèles. Sans cross validation, qui entraîne un modèle pour lequel on a optimiser les hyper paramètres.
 - main.py: met en place les pipelines avec les deux fonctions d'entrainement.

 Le dossier old_pipeline_pytorch met en place la pipeline sans cross validation. En revanche il comprend plusieurs modèles (VisionTransformer, ResNet, classicCNN, BasicTransformer).

 crop.py: permet de générer de nouvelles images en ne conservant que l'image autour des centres des fuites.
 Le résultat obetenu en l'exploitant n'est pas satisfaisant.

 Préparation des données:
 - split en test set et train set
 - split des données en validation set et en k fold pour cross validation.
 - A chaque fois, avec ou sans data augmentation sur les train set.

Types de modèle:
 - ResNet, VisionTransformer, 2 CNN basiques différents, basicTransformer

Pour la métrique, nous avons comparé les différents modèles avec leur accuracy sur test set. 

