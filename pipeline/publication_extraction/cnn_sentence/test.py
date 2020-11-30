from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report
import pathlib

from ..data_util import read_homepub, split_data


if __name__ == "__main__":
    parent_dir = pathlib.Path(__file__).resolve().parent
    pretrained_model = keras.models.load_model(parent_dir / "model", compile=False)
    samples, labels = read_homepub()
    _, (samples_test, labels_test) = split_data(samples, labels)
    X_test = [[sample] for sample in samples_test]
    y_test = np.array(labels_test)
    y_pred = np.where(pretrained_model.predict(X_test).flatten() > 0.5, 1, 0)
    print(classification_report(y_test, y_pred))
