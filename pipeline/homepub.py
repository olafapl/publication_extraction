from __future__ import annotations
import numpy as np
import pathlib
import json
import re
from typing import Tuple, Dict, List

from .data_util import data_dir


def clean_line(line: str) -> str:
    return re.sub(r"(\\|\'|\")", "", line).strip().lower()


def recall_score(tp: int, fn: int) -> float:
    return tp / (tp + fn)


def precision_score(tp: int, fp: int) -> float:
    return tp / (tp + fp)


def f1_score(precision: int, recall: int) -> float:
    return (2 * precision * recall) / (precision + recall)


class Page:
    def __init__(self, text_path: pathlib.Path, tag_path: pathlib.Path):
        self.lines = []
        self.labels = []
        self.multiline_publication_lines = []
        pub_line_nums = set()
        with tag_path.open() as tag_file:
            json_tag = json.load(tag_file)
            self.is_academic_homepage = json_tag["is_personal_homepage"] == "T"
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
                        self.lines.extend(cleaned_lines)
                        self.labels.extend([1] * len(cleaned_lines))
                        if len(cleaned_lines) > 1:
                            self.multiline_publication_lines.append(
                                list(
                                    range(
                                        len(self.lines) - len(cleaned_lines),
                                        len(self.lines),
                                    )
                                )
                            )

        with text_path.open() as text_file:
            for i, line in enumerate(text_file.readlines()):
                cleaned_line = clean_line(line)
                if i + 1 in pub_line_nums or cleaned_line in self.lines:
                    continue
                self.lines.append(cleaned_line)
                self.labels.append(0)

        self.contains_publication_list = any(label == 1 for label in self.labels)
        self.contains_multiline_publications = len(self.multiline_publication_lines) > 0


class HomePub:
    def __init__(self):
        pages: List[Page] = []
        for subdir in (data_dir / "homepub-2500").iterdir():
            tag_path = next(subdir.glob("*.tag.json"), None)
            text_path = next(subdir.glob("*.txt"), None)

            if not tag_path or not text_path:
                continue

            pages.append(Page(text_path, tag_path))

        self.split(pages)

    def split(self, pages: List[Page], test_ratio=0.4, train_val_ratio=0.2, seed=1337):
        np.random.RandomState(seed).shuffle(pages)

        num_test_samples = int(test_ratio * len(pages))
        self.train = pages[:-num_test_samples]
        self.test = pages[-num_test_samples:]

        num_val_samples = int(train_val_ratio * len(self.train))
        self.val = self.train[-num_val_samples:]
        self.train = self.train[:-num_val_samples]

    def evaluate(self, predictions: List[int], relax_matching=0.85):
        tp, tn, fp, fn = 0, 0, 0, 0
        prediction_index = 0
        for page in self.test:
            line_num = 0
            while line_num < len(page.lines):
                # If the line is part of a multi-line publication string, find the other line numbers.
                line_nums = next(
                    (
                        line_nums
                        for line_nums in page.multiline_publication_lines
                        if line_num in line_nums
                    ),
                    None,
                )

                if line_nums is None:
                    # Single-line.
                    prediction = predictions[prediction_index]
                    true = page.labels[line_num]
                    if prediction == 1:
                        if true == 1:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        if true == 1:
                            fn += 1
                        else:
                            tn += 1
                    prediction_index += 1
                    line_num += 1
                else:
                    # Multi-line.
                    preds = predictions[
                        prediction_index : (prediction_index + len(line_nums))
                    ]
                    true = page.labels[line_nums[0] : (line_nums[0] + len(line_nums))]
                    token_counts = [
                        len(line.split())
                        for line in page.lines[
                            line_nums[0] : (line_nums[0] + len(line_nums))
                        ]
                    ]
                    tp_ratio = sum(
                        pred * token_count
                        for pred, token_count in zip(preds, token_counts)
                    ) / sum(token_counts)
                    if tp_ratio >= relax_matching:
                        tp += 1
                    else:
                        fn += 1
                    prediction_index += len(line_nums)
                    line_num += len(line_nums)

        precision = precision_score(tp, fp)
        recall = recall_score(tp, fn)
        f1 = f1_score(precision, recall)
        print(f"Precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
