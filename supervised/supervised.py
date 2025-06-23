import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from Deep_Feature_Extraction import load_and_extract_features

class SupervisedLearning:
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
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_folder, filename))
                    self.labels.append(class_name)
        print(f"Total gambar yang terbaca: {len(self.image_paths)}")

    def extract_features(self):
        features = load_and_extract_features(self.image_paths)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        return features, np.array(self.labels)

    def plot_confusion_matrix(self, y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred, labels=self.class_names)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    def train_and_evaluate(self, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        print("\nEvaluasi K-Nearest Neighbor (k=3):")
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        print(classification_report(y_test, y_pred_knn))
        self.plot_confusion_matrix(y_test, y_pred_knn, "Confusion Matrix - KNN")

        print("\nEvaluasi Logistic Regression:")
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train, y_train)
        y_pred_logreg = logreg.predict(X_test)
        print(classification_report(y_test, y_pred_logreg))
        self.plot_confusion_matrix(y_test, y_pred_logreg, "Confusion Matrix - Logistic Regression")