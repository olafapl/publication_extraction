import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import pathlib
import json
import os
import re
import random
from typing import Tuple, Dict, List


def clean_line(line: str) -> str:
    line = re.sub(r"\\", "", line)
    line = re.sub(r"\'", "", line)
    line = re.sub(r"\"", "", line)
    return line.strip().lower()


def read_data(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    lines, labels = [], []
    for subdir in path.iterdir():
        tag_path = next(subdir.glob("*.tag.json"), None)
        text_path = next(subdir.glob("*.txt"), None)

        if not tag_path or not text_path:
            continue

        publication_line_nums = []
        with tag_path.open() as tag_file:
            json_tag = json.load(tag_file)

            # We're interested only in personal homepages
            if json_tag["is_personal_homepage"] == "F":
                continue

            if json_tag["contain_publication_list"] == "T":
                for publication in json_tag["publications"]:
                    line_num = publication.get("line_num")
                    if line_num:
                        publication_line_nums.extend(line_num)

        with text_path.open() as text_file:
            for i, line in enumerate(text_file.readlines()):
                lines.append(clean_line(line))
                labels.append(i + 1 in publication_line_nums)

    return np.array(lines), np.array(labels).astype("int")


def split_data(
    samples: np.ndarray, labels: np.ndarray, validation_split=0.2, seed=1337
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    samples_copy = samples.copy()
    labels_copy = labels.copy()
    np.random.RandomState(seed).shuffle(samples_copy)
    np.random.RandomState(seed).shuffle(labels_copy)

    num_validation_samples = int(validation_split * len(samples))
    samples_train = samples_copy[:-num_validation_samples]
    samples_val = samples_copy[-num_validation_samples:]
    labels_train = labels_copy[:-num_validation_samples]
    labels_val = labels_copy[-num_validation_samples:]

    print(f"{len(samples_train)} training samples, {len(samples_val)} val samples")

    return (samples_train, labels_train), (samples_val, labels_val)


def read_embeddings(path: pathlib.Path) -> Dict[str, np.ndarray]:
    """Read GloVe word embeddings from the data directory and create an embedding index."""
    embedding_index = {}
    with path.open() as embedding_file:
        for line in embedding_file:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, sep=" ")
            embedding_index[word] = coefs
    return embedding_index


def create_embedding_matrix(
    word_index: Dict[str, int],
    embedding_index: Dict[str, np.ndarray],
    num_tokens,
    embedding_dim,
):
    embedding_matrix = np.zeros([num_tokens, embedding_dim])
    hits, misses = 0, 0
    for word, i in word_index.items():
        word_embedding = embedding_index.get(word)
        if word_embedding is not None:
            embedding_matrix[i] = word_embedding
            hits += 1
        else:
            misses += 1
    print(f"Converted {hits} words ({misses} misses)")
    return embedding_matrix


if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).resolve().parent / "data"

    samples, labels = read_data(data_dir / "homepub-2500")
    (samples_train, labels_train), (samples_val, labels_val) = split_data(
        samples, labels
    )

    # Create word index.
    max_len = 200
    vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=max_len)
    vectorizer.adapt(samples_train)
    vocab = vectorizer.get_vocabulary()
    word_index = dict(zip(vocab, range(len(vocab))))
    num_tokens = len(vocab) + 2

    # Read GloVe embeddings and create embedding matrix.
    embedding_dim = 300
    embedding_index = read_embeddings(data_dir / "glove" / "glove.6B.300d.txt")
    embedding_matrix = create_embedding_matrix(
        word_index, embedding_index, num_tokens, embedding_dim
    )

    # Build the model.
    int_sequences_input = keras.Input(shape=(max_len,), dtype="int64")

    # Embedding layer.
    x = layers.Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )(int_sequences_input)

    # Convolution layers with three filter sizes and L2 regularization.
    filter_sizes = [3, 4, 5]
    num_filters = 100
    conv_layers = [
        layers.Conv1D(
            num_filters,
            filter_size,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(3),
        )(x)
        for filter_size in filter_sizes
    ]
    x = layers.concatenate(conv_layers, axis=1)

    # Max-over-time pooling layer.
    x = layers.GlobalMaxPool1D()(x)

    # Fully connected layer with dropout and softmax output.
    x = layers.Dense(num_filters, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(int_sequences_input, x)
    model.summary()

    X_train = vectorizer(np.array([[s] for s in samples_train])).numpy()
    X_val = vectorizer(np.array([[s] for s in samples_val])).numpy()
    y_train = np.array(labels_train)
    y_val = np.array(labels_val)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    model.fit(X_train, y_train, batch_size=50, epochs=3, validation_data=(X_val, y_val))
