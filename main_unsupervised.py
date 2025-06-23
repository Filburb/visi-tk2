from unsupervised.unsupervised import UnsupervisedLearning

unsup = UnsupervisedLearning("00000")
unsup.load_dataset()  # Panggil method yang sudah ada

features = unsup.extract_features()
reduced = unsup.apply_pca(features)
cluster_labels = unsup.apply_kmeans(reduced, n_clusters=4)
unsup.plot_clusters(reduced, cluster_labels)
unsup.count_cluster_labels(cluster_labels)
