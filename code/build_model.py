
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # SVM classifier
from sklearn.metrics import classification_report
# Importation des bibliothèques nécessaires
import cv2  # Bibliothèque pour la vision par ordinateur (OpenCV)
import numpy as np  # Bibliothèque pour la manipulation de tableaux et calculs scientifiques
from sklearn.cluster import KMeans  # Pour appliquer l'algorithme KMeans (non utilisé dans ce code, probablement résidu)
from sklearn.metrics.pairwise import euclidean_distances  # Pour calculer les distances euclidiennes entre vecteurs
import os  # Pour la gestion des fichiers et répertoires
import pickle  # Pour sauvegarder et charger des objets Python dans des fichiers
import pandas as pd  # Pour manipuler des données tabulaires via des DataFrames


# Fonction pour convertir les chaînes en tableau NumPy
def convert_to_numpy_array(data_str):
    # Remplacement des crochets et conversion en tableau numpy
    data_str = data_str.replace('[', '').replace(']', '').replace('  ', ' ')
    return np.fromstring(data_str, sep=' ')

# 1. Charger le dataset
df = pd.read_csv('data.csv')

# 2. Préparation des données
# Convertir la colonne 'caractéristique' de string en tableau numpy
# Appliquer la fonction à chaque élément de la colonne 'caractéristique'
df['caractéristique'] = df['caractéristique'].apply(convert_to_numpy_array)


# Récupérer les caractéristiques (features) et les classes (labels)
X = np.stack(df['caractéristique'].values)  # Convertir les listes en un tableau numpy
y = df['classe'].values  # Les classes

# Diviser les données en ensemble d'entraînement et ensemble de test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entraîner le modèle SVM
svm = SVC(kernel='linear')  # Utilisation d'un SVM avec un noyau linéaire
svm.fit(X_train, y_train)  # Entraînement sur les données

# 4. Évaluer le modèle
y_pred = svm.predict(X_test)  # Prédire les classes pour l'ensemble de test
print(y_pred)
print(classification_report(y_test, y_pred))  # Afficher le rapport de classification


# Fonction pour calculer l'histogramme des descripteurs SIFT d'une image en utilisant un codebook
def compute_histogram(image, codebook):
    sift = cv2.SIFT_create()  # Création d'un détecteur SIFT (Scale-Invariant Feature Transform)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Conversion de l'image en niveaux de gris
    
    # Extraction des points clés et des descripteurs SIFT de l'image
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    if descriptors is None:  # Si aucun descripteur n'est trouvé dans l'image
        return np.zeros(len(codebook))  # Retourne un histogramme vide (zéros) de la taille du codebook

    # Initialisation de l'histogramme avec des zéros (de même taille que le codebook)
    histogram = np.zeros(len(codebook))
    
    # Pour chaque descripteur SIFT, on trouve le cluster le plus proche dans le codebook
    for descriptor in descriptors:
        distances = np.linalg.norm(codebook - descriptor, axis=1)  # Calcul des distances euclidiennes
        best_match = np.argmin(distances)  # Trouver l'indice du cluster le plus proche
        histogram[best_match] += 1  # Incrémenter la fréquence du cluster correspondant dans l'histogramme

    # Normalisation de l'histogramme pour que la somme des valeurs fasse 1 (normalisation L1)
    histogram /= np.sum(histogram)

    return histogram  # Retourne l'histogramme normalisé



# Charger le codebook (dictionnaire de clusters SIFT)
print("chargement du codebook ..")
with open("codebook.pkl", "rb") as fichier:  # Ouverture du fichier contenant le codebook en mode binaire
    codebook = pickle.load(fichier)  # Chargement du codebook

zip_folder="./ref"
for filename in os.listdir(zip_folder):
    path_image_ref = os.path.join(zip_folder, filename)
    # Charger l'image de référence pour la comparaison
    print("chargement de l'image de reférence ..")
    image_ref = cv2.imread(path_image_ref)  # Lecture de l'image
    histogram_image_ref = compute_histogram(image_ref, codebook)  # Calcul de l'histogramme SIFT pour l'image de référence


    pred = svm.predict([histogram_image_ref,histogram_image_ref])  # Prédire les classes pour l'ensemble de test
    print(pred)