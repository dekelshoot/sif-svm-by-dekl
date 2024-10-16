# Importation des bibliothèques nécessaires
import cv2  # Bibliothèque pour la vision par ordinateur (OpenCV)
import numpy as np  # Bibliothèque pour la manipulation de tableaux et calculs scientifiques
from sklearn.cluster import KMeans  # Pour appliquer l'algorithme KMeans (non utilisé dans ce code, probablement résidu)
from sklearn.metrics.pairwise import euclidean_distances  # Pour calculer les distances euclidiennes entre vecteurs
import os  # Pour la gestion des fichiers et répertoires
import pickle  # Pour sauvegarder et charger des objets Python dans des fichiers
import pandas as pd  # Pour manipuler des données tabulaires via des DataFrames
from sklearn.preprocessing import MinMaxScaler
# Fonction pour calculer l'histogramme des descripteurs SIFT d'une image en utilisant un codebook
def compute_histogram(image, codebook):
    sift = cv2.SIFT_create()  # Création d'un détecteur SIFT (Scale-Invariant Feature Transform)

    # Extraction des points clés et des descripteurs SIFT de l'image
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
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

def create_feature_bow(image, BoW, num_cluster):
        sift = cv2.SIFT_create()  # Création d'un détecteur SIFT (Scale-Invariant Feature Transform)
        # Extraction des points clés et des descripteurs SIFT de l'image
        keypoints, descriptors = sift.detectAndCompute(image, None)
        features = np.array([0] * num_cluster)

        if descriptors is not None:
            distance = euclidean_distances(descriptors, BoW)

            argmin = np.argmin(distance, axis = 1)

            for j in argmin:
                features[j] += 1
        return features
# Charger le codebook (dictionnaire de clusters SIFT)
print("chargement du codebook ..")
with open("codebook.pkl", "rb") as fichier:  # Ouverture du fichier contenant le codebook en mode binaire
    codebook = pickle.load(fichier)  # Chargement du codebook

# Charger l'image de référence pour la comparaison
print("chargement de l'image de reférence ..")
path_image_ref = "./ref/1.png"  # Chemin de l'image de référence
image_ref = cv2.imread(path_image_ref)  # Lecture de l'image
histogram_image_ref = create_feature_bow(image_ref, codebook,150)  # Calcul de l'histogramme SIFT pour l'image de référence


# Dossier contenant les images à comparer
folder = './img3'
seuil = 0.5  # Seuil pour classifier les images
data = {}  # Initialisation du dictionnaire pour stocker les résultats

# Boucle sur tous les fichiers du dossier contenant les images
for filename in os.listdir(folder):
    img_path = os.path.join(folder, filename)  # Chemin complet de l'image
    image = cv2.imread(img_path)  # Lecture de l'image
    histogram_image = create_feature_bow(image, codebook,150)  # Calcul de l'histogramme pour l'image courante
    
    # Calculer la distance euclidienne entre l'image de référence et l'image courante
    distance = euclidean_distances(histogram_image_ref, histogram_image)
    # Normaliser entre 0 et 1
    scaler = MinMaxScaler()
    print(distance)
    distance = scaler.fit_transform(distance)
    print(distance)
    # Si la distance est supérieure ou égale au seuil, classer l'image dans la classe 1, sinon dans la classe 0
    if distance[0][0] >= seuil:
        data[filename] = {
            'image': filename,
            'caractéristique': histogram_image,
            'classe': 1  # Classe 1 si la distance est au-dessus du seuil
        }
    else:
        data[filename] = {
            'image': filename,
            'caractéristique': histogram_image,
            'classe': 0  # Classe 0 sinon
        }


print("transformation en dataset ....")
# Création d'un dataset sous forme de DataFrame à partir du dictionnaire "data"
datas = {
    'image': [d['image'] for d in data.values()],  # Récupère les noms des images
    'caractéristique': [d['caractéristique'] for d in data.values()],  # Récupère les caractéristiques (histogrammes)
    'classe': [d['classe'] for d in data.values()]  # Récupère les classes associées
}

df = pd.DataFrame(datas)  # Création du DataFrame à partir des données

print("sauvegarde des données ...")
# Sauvegarde du DataFrame en fichier CSV
df.to_csv('data.csv', index=False)  # Sauvegarde du DataFrame dans un fichier CSV, sans les index

print("Le fichier CSV a été sauvegardé avec succès.")  # Message de confirmation
