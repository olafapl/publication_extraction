import pycrfsuite
from sklearn.metrics import classification_report
import pathlib
import re
import argparse

from .train import sentence2features, sentence2labels, sentence2tokens, model_path
from ..umass import read_umass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="filename (without extension) containing a trained model",
    )
    args = parser.parse_args()

    tagger = pycrfsuite.Tagger()
    tagger.open(model_path(args.model))
    train, val, test = read_umass()
    X_test = [sentence2features(sentence) for sentence in test]
    y_test = [sentence2labels(sentence) for sentence in test]
    y_pred = [tagger.tag(sentence) for sentence in X_test]
    print(
        classification_report(
            [label for sentence in y_test for label in sentence],
            [label for sentence in y_pred for label in sentence],
            digits=4,
        )
    )
