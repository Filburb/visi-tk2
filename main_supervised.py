
from supervised.supervised import SupervisedLearning

DATASET_PATH = "00000"  # Ganti sesuai dengan path dataset

model = SupervisedLearning(DATASET_PATH)
model.load_dataset()
features, labels = model.extract_features()
model.train_and_evaluate(features, labels)