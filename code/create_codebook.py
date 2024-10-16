import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import pickle
def create_codebook(images, k=100):
    sift = cv2.SIFT_create()
    descriptors_list = []

    # Extraction des points clés et des descripteurs SIFT pour chaque image
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)

    # Concatenation des descripteurs pour toutes les images
    all_descriptors = np.vstack(descriptors_list)

    # Application de K-means pour créer le codebook
    kmeans = KMeans(n_clusters=k, random_state=0).fit(all_descriptors)

    # Les centres des clusters deviennent notre codebook
    codebook = kmeans.cluster_centers_
    return codebook

# Charger les images et créer le codebook
folder = './img3'
images = []
print("chargement d'images...")
for filename in os.listdir(folder):
  img_path = os.path.join(folder, filename)
  img = cv2.imread(img_path)
  images.append(img)


print("Création du notebook...")
k = 150  # Nombre de clusters pour K-means
codebook = create_codebook(images, k=k)

# Sauvegarde du codebook
print("Sauvegarde du codebook ...")
with open("codebook.pkl", "wb") as fichier:  # 'wb' signifie écriture en mode binaire
    pickle.dump(codebook, fichier)

print("sauvegarde réussit ...")