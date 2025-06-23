import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from Deep_Feature_Extraction import load_and_extract_features

class UnsupervisedLearning:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.image_paths = []
        self.labels = []
        self.class_names = []

    def load_dataset(self):
        for class_name in sorted(os.listdir(self.dataset_path)):
            class_folder = os.path.join(self.dataset_path, class_name)
            if not os.path.isdir(class_folder):
                continue
            self.class_names.append(class_name)
            for filename in os.listdir(class_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_folder, filename))
                    self.labels.append(class_name)
        print(f"Total gambar yang terbaca: {len(self.image_paths)}")

    def extract_features(self):
        features = load_and_extract_features(self.image_paths)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        return features_scaled

    def apply_pca(self, features, n_components=0.95):
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(features)
        return reduced

    def apply_kmeans(self, features, n_clusters=4):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(features)
        return cluster_labels

    def count_cluster_labels(self, cluster_labels):
        cluster_info = defaultdict(lambda: defaultdict(int))
        for cluster_id, true_label in zip(cluster_labels, self.labels):
            cluster_info[cluster_id][true_label] += 1

        print("\nDistribusi label sebenarnya di tiap cluster:\n")
        for cluster_id, label_count in sorted(cluster_info.items()):
            print(f"Cluster {cluster_id}:")
            for label, count in sorted(label_count.items()):
                print(f"  {label}: {count} gambar")
            print()

    def plot_clusters(self, reduced_data, cluster_labels):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='tab10')
        plt.title("K-Means Clustering with PCA")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.grid(True)
        plt.colorbar(scatter)
        plt.show()
