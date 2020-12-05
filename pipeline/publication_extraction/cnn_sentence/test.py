from tensorflow import keras
import numpy as np
import pathlib
import argparse

from ...homepub import HomePub


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN-Sentence test")
    parser.add_argument(
        "model",
        type=str,
        help="folder containing trained model inside the /models directory",
    )
    args = parser.parse_args()
    parent_dir = pathlib.Path(__file__).resolve().parent
    pretrained_model = keras.models.load_model(
        parent_dir / "models" / args.model, compile=False
    )
    data = HomePub()
    samples_test = [sample for page in data.test for sample in page.lines]
    X_test = [[sample] for sample in samples_test]
    y_pred = np.where(pretrained_model.predict(X_test).flatten() > 0.5, 1, 0).tolist()
    print("Exact matching:")
    data.evaluate(y_pred.tolist(), relax_matching=1)
    print("Relaxed matching (85%):")
    data.evaluate(y_pred.tolist(), relax_matching=0.85)
