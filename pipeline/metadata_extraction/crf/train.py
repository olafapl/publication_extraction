import re
import string
import pycrfsuite
import argparse
import pathlib
import os
import math
from typing import List, Tuple

from ..umass import read_umass


def model_path(model: str) -> str:
    model_dir = pathlib.Path(__file__).resolve().parent / "models"
    if not model_dir.exists():
        model_dir.mkdir()
    return os.fspath(model_dir / f"{model}.crfsuite")


def is_page(word: str) -> bool:
    return bool(re.search(r"^(pages?|pp?.?)$", word.lower()))


def replace_numbers(word: str, sub="NUM") -> str:
    return re.sub(r"\d+", sub, word)


def is_digit_paren(word: str) -> bool:
    return bool(re.search(r"\(\d+\)", word))


def contains_digit(word: str) -> bool:
    return bool(re.search(r".*\d.*", word))


def ends_with_digit(word: str) -> bool:
    return bool(re.search(r".*\d$", word))


def is_range(word: str) -> bool:
    return bool(re.search(r"\d[\-––—]+\d", word))


def is_single_char_period(word: str) -> bool:
    return bool(re.search(r"^[A-Z]\.$", word))


def word2features(sentence: str, index: int) -> List[str]:
    word = sentence[index][0]
    features = [
        f"word={word}",
        f"word.lower={replace_numbers(word.lower())}",
        f"word.upper={replace_numbers(word.upper())}",
        f"word[:3]={word[:3]}",
        f"word.ispage={is_page(word)}",
        f"word.single_char={len(word) == 1}",
        f"word.istitle={word.istitle()}",
        f"word.isupper={word.isupper()}"
        f"word.single_char_period={is_single_char_period(word)}",
        f"word.isdigit={word.isdigit()}",
        f"word.isdigit_paren={is_digit_paren(word)}",
        f"word.contains_digit={contains_digit(word)}",
        f"word.ends_digit={ends_with_digit(word)}",
        f"word.ispunct={word in string.punctuation}",
        f"word.contains_dash={'-' in word}",
        f"word.contains_period={'.' in word}",
        f"word.isrange={is_range(word)}",
        f"word.num_digits={sum(c.isdigit() for c in word)}",
        f"word.num_alpha={sum(c.isalnum() for c in word)}",
        f"word.location={math.ceil(index / len(sentence) * 12)}",
    ]

    if index > 0:
        prev_word = sentence[index - 1][0]
        features.extend(
            [
                f"prev_word={prev_word}",
                f"prev_word.lower={replace_numbers(prev_word.lower())}",
                f"prev_word.upper={replace_numbers(prev_word.upper())}",
                f"prev_word[:3]={prev_word[:3]}",
                f"prev_word.ispage={is_page(prev_word)}",
                f"prev_word.single_char={len(prev_word) == 1}",
                f"prev_word.istitle={prev_word.istitle()}",
                f"prev_word.isupper={prev_word.isupper()}"
                f"prev_word.single_char_period={is_single_char_period(prev_word)}",
                f"prev_word.isdigit={prev_word.isdigit()}",
                f"prev_word.isdigit_paren={is_digit_paren(prev_word)}",
                f"prev_word.contains_digit={contains_digit(prev_word)}",
                f"prev_word.ends_digit={ends_with_digit(prev_word)}",
                f"prev_word.ispunct={prev_word in string.punctuation}",
                f"prev_word.contains_dash={'-' in prev_word}",
                f"prev_word.contains_period={'.' in prev_word}",
                f"prev_word.isrange={is_range(prev_word)}",
                f"prev_word.num_digits={sum(c.isdigit() for c in prev_word)}",
                f"prev_word.num_alpha={sum(c.isalnum() for c in prev_word)}",
            ]
        )

    if index < len(sentence) - 1:
        next_word = sentence[index + 1][0]
        features.extend(
            [
                f"next_word={next_word}",
                f"next_word.lower={replace_numbers(next_word.lower())}",
                f"next_word.upper={replace_numbers(next_word.upper())}",
                f"next_word[:3]={next_word[:3]}",
                f"next_word.ispage={is_page(next_word)}",
                f"next_word.single_char={len(next_word) == 1}",
                f"next_word.istitle={next_word.istitle()}",
                f"next_word.isupper={next_word.isupper()}"
                f"next_word.single_char_period={is_single_char_period(next_word)}",
                f"next_word.isdigit={next_word.isdigit()}",
                f"next_word.isdigit_paren={is_digit_paren(next_word)}",
                f"next_word.contains_digit={contains_digit(next_word)}",
                f"next_word.ends_digit={ends_with_digit(next_word)}",
                f"next_word.ispunct={next_word in string.punctuation}",
                f"next_word.contains_dash={'-' in next_word}",
                f"next_word.contains_period={'.' in next_word}",
                f"next_word.isrange={is_range(next_word)}",
                f"next_word.num_digits={sum(c.isdigit() for c in next_word)}",
                f"next_word.num_alpha={sum(c.isalnum() for c in next_word)}",
            ]
        )

    return features


def sentence2features(sentence: List[Tuple[str, str]]) -> List[str]:
    return [word2features(sentence, i) for i in range(len(sentence))]


def sentence2labels(sentence: List[Tuple[str, str]]) -> List[str]:
    return [label for _, label in sentence]


def sentence2tokens(sentence: List[Tuple[str, str]]) -> List[str]:
    return [token for token, _ in sentence]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        type=str,
        help="output filename (without extension) for the trained model",
    )
    args = parser.parse_args()

    train, _, __ = read_umass()

    X_train = [sentence2features(sentence) for sentence in train]
    y_train = [sentence2labels(sentence) for sentence in train]

    trainer = pycrfsuite.Trainer()
    for sentence, label in zip(X_train, y_train):
        trainer.append(sentence, label)
    trainer.set_params(
        {
            "c1": 1.0,
            "c2": 1e-3,
            "max_iterations": 50,
            "feature.possible_transitions": True,
        }
    )
    trainer.train(model_path(args.output))
