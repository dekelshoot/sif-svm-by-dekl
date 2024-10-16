# Prédiction du Cancer de la Prostate à l'aide de SIFT et SVM

Ce projet utilise des techniques avancées de vision par ordinateur et d'apprentissage automatique pour prédire le cancer de la prostate à partir d'images médicales. En combinant l'extraction de caractéristiques SIFT (Scale-Invariant Feature Transform) avec un modèle de classification SVM (Support Vector Machine), nous proposons un système capable de distinguer différents types de tissus prostatiques (normal, bénin, malin).

## Fonctionnalités

- **Extraction de caractéristiques SIFT** : Détecte les points d'intérêt dans les images médicales et extrait des descripteurs robustes.
- **Codebook par K-Means** : Regroupe les descripteurs SIFT en un dictionnaire de visuels pour améliorer l'efficacité de la classification.
- **Classification SVM** : Modèle linéaire de classification pour prédire la classe du tissu (normal, bénin, malin) à partir des descripteurs d'image.
- **Évaluation des performances** : Génère un rapport de classification pour évaluer la précision, la sensibilité et la spécificité du modèle.

## Structure du projet

- **`buid_mode.py`** : Script principal pour l'entraînement du modèle SVM et l'évaluation des performances.
- **`classification.py`** : Permet de comparer des images médicales via la distance euclidienne des descripteurs SIFT.
- **`creat_codebook.py`** : Génère un codebook SIFT via l'algorithme K-Means pour grouper les descripteurs.
- **`test.py`** : Teste la précision du modèle sur un jeu d'images de référence et produit des résultats de classification.

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/username/prostate-cancer-prediction.git
   ```
2. Installez les dépendances requises :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

1. **Génération du codebook** :
   - Exécutez `creat_codebook.py` pour créer un dictionnaire de clusters SIFT à partir de vos images.
2. **Entraînement et évaluation** :
   - Lancez `buid_mode.py` pour entraîner le modèle SVM et générer un rapport d'évaluation.
3. **Test et comparaison** :
   - Utilisez `test.py` pour tester la précision du modèle sur de nouvelles images et vérifier les prédictions.

## Contributeurs

- **Ton nom** - Développeur principal
- **Mr. Volt** - Collaborateur

---

Ce projet est un outil de recherche en vue d'améliorer la détection et la classification des tissus prostatiques à partir d'images médicales. Il n'est pas destiné à un usage clinique sans validation médicale approfondie.
