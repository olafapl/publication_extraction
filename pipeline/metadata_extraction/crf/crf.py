import pycrfsuite
import pathlib
import re
from typing import List

from .train import sentence2features, model_path
from ..metadata_extractor import MetadataExtractor, Publication
from ..umass import tag_label_map


def tokenize(publication: str) -> List[str]:
    augmented = "".join([c if c != "," else " , " for c in publication])
    return augmented.split()


class CRF(MetadataExtractor):
    def __init__(self, model: str):
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(model_path(model))

    def parse_publications(
        self,
        publication_strings: List[str],
    ) -> List[Publication]:
        publications = []
        for publication_string in publication_strings:
            metadata = {}
            tokens = tokenize(publication_string)
            predictions: List[str] = self.tagger.tag(sentence2features(tokens))
            for token, tag in zip(tokens, predictions):
                if tag.startswith("B-"):
                    label = tag_label_map[tag[2:]]
                    if label not in metadata:
                        metadata[label] = []
                    metadata[label].append(token)
                elif tag.startswith("I-"):
                    label = tag_label_map[tag[2:]]
                    if label not in metadata:
                        metadata[label] = []
                        metadata[label].append(token)
                    else:
                        metadata[label][-1] += f" {token}"
            publications.append(
                Publication(
                    authors=metadata.get("person", None),
                    title=metadata.get("title", None),
                    year=metadata.get("year", None),
                    journal=metadata.get("journal", None),
                )
            )
        return publications
