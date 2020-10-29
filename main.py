from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report
import pathlib
import src.cnn_sentence as cnn


if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).resolve().parent
    pretrained_model = keras.models.load_model(
        current_dir / "pretrained", compile=False
    )
    samples, labels = cnn.read_homepub(current_dir / "data" / "homepub-2500")
    _, (samples_test, labels_test) = cnn.split_data(samples, labels)
    X_test = [[sample] for sample in samples_test]
    y_test = np.array(labels_test)
    y_pred = np.where(pretrained_model.predict(X_test).flatten() > 0.5, 1, 0)
    print(classification_report(y_test, y_pred))
