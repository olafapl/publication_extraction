from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import pathlib
import json
import re
from typing import Tuple, Dict, List


def clean_line(line: str) -> str:
    return re.sub(r"(\\|\'|\")", "", line).strip().lower()


def read_dataset(path: pathlib.Path) -> Tuple[List[str], List[int]]:
    """Read the HomePub dataset.

    Args:
        path (pathlib.Path): Data path.

    Returns: Tuple[List[str], List[int]]: Tuple (lines, labels), where a 1 label signifies that a
        line is a publication string. Lines comprising multi-line publication strings are
        concatenated to single lines.
    """
    lines, labels = [], []
    for subdir in path.iterdir():
        tag_path = next(subdir.glob("*.tag.json"), None)
        text_path = next(subdir.glob("*.txt"), None)

        if not tag_path or not text_path:
            continue

        pub_line_nums: List[List[int]] = []
        with tag_path.open() as tag_file:
            json_tag = json.load(tag_file)

            # We're interested only in personal homepages
            if json_tag["is_personal_homepage"] == "F":
                continue

            if json_tag["contain_publication_list"] == "T":
                for publication in json_tag["publications"]:
                    line_num = publication.get("line_num")
                    if line_num:
                        pub_line_nums.append(line_num)

        with text_path.open() as text_file:
            next_pub_line_num = None
            for i, line in enumerate(text_file.readlines()):
                line_num = i + 1
                cleaned_line = clean_line(line)

                if not next_pub_line_num or max(next_pub_line_num) < line_num:
                    next_pub_line_num = (
                        pub_line_nums.pop(0) if len(pub_line_nums) else None
                    )

                if next_pub_line_num and line_num >= min(next_pub_line_num):
                    # The current line is part of a publication string.
                    if line_num > min(next_pub_line_num):
                        # The line is part of a multi-line publication string, but not the first
                        # line.
                        lines[-1] = lines[-1] + cleaned_line
                    else:
                        # The line is the first (and perhaps only) line of a publication string.
                        lines.append(cleaned_line)
                        labels.append(1)
                else:
                    lines.append(cleaned_line)
                    labels.append(0)

    return lines, labels


def split_data(
    samples: List, labels: List, validation_split=0.2, seed=1337
) -> Tuple[Tuple[List, List], Tuple[np.ndarray, np.ndarray]]:
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
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"

    samples, labels = read_dataset(data_dir / "homepub-2500")
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

    model = cnn_sentence(max_len, num_tokens, embedding_matrix, embedding_dim)

    X_train = vectorizer([[s] for s in samples_train]).numpy()
    X_val = vectorizer([[s] for s in samples_val]).numpy()
    y_train = np.array(labels_train)
    y_val = np.array(labels_val)

    model.fit(X_train, y_train, batch_size=50, epochs=3, validation_data=(X_val, y_val))
