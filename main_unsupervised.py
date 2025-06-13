from unsupervised.unsupervised import UnsupervisedLearning

unsup = UnsupervisedLearning("00000")
unsup.load_images()

pca_result = unsup.apply_pca()
kmeans_labels = unsup.apply_kmeans(n_clusters=3)
unsup.plot_clusters(pca_result, kmeans_labels, title="K-Means Clustering with PCA")
