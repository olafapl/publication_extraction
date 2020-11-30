from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import pathlib
import json
import re
from typing import Tuple, Dict, List

from ..data_util import read_homepub, split_data, data_dir


def read_embeddings(path: pathlib.Path) -> Dict[str, np.ndarray]:
    """Read pretrained GloVe word embeddings and create an embedding index.

    Args:
        path (pathlib.Path): Word vector path.

    Returns:
        Dict[str, np.ndarray]: Word embedding index.
    """
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
) -> np.ndarray:
    embedding_matrix = np.zeros([num_tokens, embedding_dim])
    for word, i in word_index.items():
        word_embedding = embedding_index.get(word)
        if word_embedding is not None:
            embedding_matrix[i] = word_embedding
    return embedding_matrix


def cnn_sentence(
    max_len: int,
    num_tokens: int,
    embedding_matrix: np.ndarray,
    embedding_dim: int,
    filter_sizes=[3, 4, 5],
    filter_num=100,
    dropout_rate=0.5,
    l2_constraint=3.0,
) -> keras.Model:
    """Build and compile a CNN-Sentence model.

    Args:
        max_len (int): Max input (line) length.
        num_tokens (int): Vocabulary size.
        embedding_matrix (np.ndarray): Word embedding matrix.
        embedding_dim (int): Word embedding dimensionality.
        filter_sizes (list, optional): Filter sizes. Defaults to [3, 4, 5].
        filter_num (int, optional): Number of filters per filter size. Defaults to 100.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.5.
        l2_constraint (float, optional): L2 constraint on penultimate layer.

    Returns:
        keras.Model: Compiled model.
    """
    int_sequences_input = keras.Input(shape=(max_len,), dtype="int64")

    # Embedding layer.
    x = layers.Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )(int_sequences_input)

    # Convolution layers with three filter sizes.
    conv_layers = [
        layers.Conv1D(
            filter_num,
            filter_size,
            activation="relu",
        )(x)
        for filter_size in filter_sizes
    ]
    x = layers.concatenate(conv_layers, axis=1)

    # Max-over-time pooling layer.
    x = layers.GlobalMaxPool1D()(x)

    # Fully connected layer with dropout and L2 constraint.
    x = layers.Dense(
        filter_num,
        activation="relu",
        kernel_constraint=keras.constraints.MaxNorm(l2_constraint),
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(int_sequences_input, x)
    model.summary()
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            "acc",
            keras.metrics.Precision(name="prec"),
            keras.metrics.Recall(name="rec"),
        ],
    )

    return model


if __name__ == "__main__":
    parent_dir = pathlib.Path(__file__).resolve().parent

    # Create training, testing, and validation splits. We split the data 60/40 between training and
    # testing and use 20% of the training data for validation.
    samples, labels = read_homepub()
    (samples_train, labels_train), (samples_test, labels_test) = split_data(
        samples, labels
    )
    (samples_train, labels_train), (samples_val, labels_val) = split_data(
        samples_train, labels_train, split=0.2
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

    # Create and train model.
    model = cnn_sentence(max_len, num_tokens, embedding_matrix, embedding_dim)

    X_train = vectorizer([[s] for s in samples_train]).numpy()
    X_val = vectorizer([[s] for s in samples_val]).numpy()
    y_train = np.array(labels_train)
    y_val = np.array(labels_val)

    model.fit(
        X_train,
        y_train,
        batch_size=50,
        epochs=30,
        validation_data=(X_val, y_val),
        class_weight={0: 1, 1: np.sum(y_train == 0) / np.sum(y_train == 1)},
        callbacks=[
            keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
        ],
    )

    # Create and save end-to-end model.
    string_input = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(string_input)
    x = model(x)
    end_to_end_model = keras.Model(string_input, x)
    end_to_end_model.save(parent_dir / "models" / "model")
