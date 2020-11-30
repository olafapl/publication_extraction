import numpy as np
import pathlib
import json
import re
from typing import Tuple, Dict, List

data_dir = pathlib.Path(__file__).resolve().parent / ".." / ".." / "data"


def clean_line(line: str) -> str:
    return re.sub(r"(\\|\'|\")", "", line).strip().lower()


def read_homepub() -> Tuple[List[str], List[int]]:
    """Read the HomePub dataset.

    Args:
        path (pathlib.Path): Data path.

    Returns: Tuple[List[str], List[int]]: Tuple (lines, labels), where a 1 label signifies that a
        line is (part of) a publication string.
    """
    lines, labels = [], []
    for subdir in (data_dir / "homepub-2500").iterdir():
        tag_path = next(subdir.glob("*.tag.json"), None)
        text_path = next(subdir.glob("*.txt"), None)

        if not tag_path or not text_path:
            continue

        pub_line_nums = set()
        pub_lines = set()
        with tag_path.open() as tag_file:
            json_tag = json.load(tag_file)

            if json_tag["contain_publication_list"] == "T":
                for publication in json_tag["publications"]:
                    if (
                        "start_index_in_file" in publication
                        or "line_num" in publication
                        or "length" in publication
                    ):
                        pub_line_nums.update(publication.get("line_num", []))
                        cleaned_lines = [
                            clean_line(line) for line in publication["text"].split("\n")
                        ]
                        pub_lines.update(cleaned_lines)
                        lines.extend(cleaned_lines)
                        labels.extend([1] * len(cleaned_lines))

        with text_path.open() as text_file:
            for i, line in enumerate(text_file.readlines()):
                cleaned_line = clean_line(line)
                if i + 1 in pub_line_nums or cleaned_line in pub_lines:
                    continue
                lines.append(cleaned_line)
                labels.append(0)

    return lines, labels


def split_data(
    samples: List, labels: List, split=0.4, seed=1337
) -> Tuple[Tuple[List, List], Tuple[np.ndarray, np.ndarray]]:
    samples_copy = samples.copy()
    labels_copy = labels.copy()
    np.random.RandomState(seed).shuffle(samples_copy)
    np.random.RandomState(seed).shuffle(labels_copy)

    num_validation_samples = int(split * len(samples))
    samples_train = samples_copy[:-num_validation_samples]
    samples_val = samples_copy[-num_validation_samples:]
    labels_train = labels_copy[:-num_validation_samples]
    labels_val = labels_copy[-num_validation_samples:]

    return (samples_train, labels_train), (samples_val, labels_val)
