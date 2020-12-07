import re
import pathlib
from typing import List, Tuple

from ..data_util import data_dir

Sentence = List[Tuple[str, str]]


def label_value(label: str) -> str:
    return re.findall(r"</?(.*)>", label)[0]


label_tag_map = {
    "person": "AUTH",
    "title": "TITLE",
    "year": "YEAR",
    "journal": "JOURN",
}


tag_label_map = {tag: label for label, tag in label_tag_map.items()}


def parse_line(line: str) -> Sentence:
    tokens = []
    current_tag = "O"
    for token in line.split():
        if re.search(r"</(.*)>", token):
            label = label_value(token)
            if label in label_tag_map and current_tag.endswith(label_tag_map[label]):
                current_tag = "O"
        elif re.search(r"<(.*)>", token):
            label = label_value(token)
            if label in label_tag_map:
                current_tag = f"B-{label_tag_map[label]}"
        else:
            tokens.append((token, current_tag))
            if current_tag.startswith("B-"):
                current_tag = f"I-{current_tag[2:]}"
    return tokens


def read_file(filename: str) -> List[Sentence]:
    with (data_dir / "umass" / filename).open() as file:
        return [parse_line(line) for line in file.readlines()]


def read_umass() -> Tuple[List[Sentence], List[Sentence], List[Sentence]]:
    return [
        read_file("training.docs"),
        read_file("dev.docs"),
        read_file("testing.docs"),
    ]
