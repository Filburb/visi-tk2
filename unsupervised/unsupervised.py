import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter

class UnsupervisedLearning:
    def __init__(self, image_folder, img_size=(64, 64)):
        self.image_folder = image_folder
        self.img_size = img_size
        self.images = []
        self.image_names = []
        self.features = None

    def load_images(self):
        self.images = []
        self.image_names = []
        for file in os.listdir(self.image_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(self.image_folder, file)
                img = cv2.imread(path)
                img = cv2.resize(img, self.img_size)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                flat = gray.flatten()
                self.images.append(flat)
                self.image_names.append(file)
        self.features = np.array(self.images)

    def apply_kmeans(self, n_clusters=3):
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(self.features)
        return labels

    def apply_pca(self, n_components=2):
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(self.features)
        return reduced

    def plot_clusters(self, reduced_data, labels, title='Cluster Result'):
        plt.figure(figsize=(8,6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
        plt.title(title)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.grid(True)
        plt.show()

    def count_cluster_labels(self, labels):
        label_counts = Counter(labels)
        print("Jumlah gambar di tiap cluster:")
        for cluster_id, count in label_counts.items():
            print(f"Cluster {cluster_id}: {count} gambar")
