import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Fonction pour extraire les caractéristiques SIFT d'une image
def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

# Charger les images et les étiquettes
def load_images_and_labels(image_paths, labels):
    features = []
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        descriptors = extract_sift_features(image)
        if descriptors is not None:
            features.append(descriptors)
    return np.array(features), np.array(labels)

# Classification SVM
def train_svm(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Exécuter le modèle
if __name__ == "__main__":
    # Remplacez par les chemins de vos images et leurs étiquettes correspondantes
    image_paths = ['./ref/1.png', './ref/2.png', './ref/3.png']
    labels = [0, 1, 2]  # 0: Normal, 1: Bénin, 2: Malin

    # Charger les images et extraire les caractéristiques
    features, labels = load_images_and_labels(image_paths, labels)

    # Entraîner et évaluer le modèle SVM
    accuracy = train_svm(features, labels)
    print(f"Précision du modèle : {accuracy * 100:.2f}%")
