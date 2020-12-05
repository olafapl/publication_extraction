from tensorflow import keras
import numpy as np
import pathlib
from typing import List

from ..publication_extractor import PublicationExtractor


class CNNSentence(PublicationExtractor):
    def __init__(self, model: str):
        """
        Args:
            model (str): Name of directory (inside "/models") to load the model from.
        """
        super().__init__()
        model_dir = pathlib.Path(__file__).resolve().parent / "models" / model
        self.model = keras.models.load_model(model_dir, compile=False)

    def extract_publications(self, text: str, source: str) -> List[str]:
        lines = text.split("\n")
        predictions = self.model.predict([[line] for line in lines]).flatten()
        return np.array(lines)[predictions > 0.5].tolist()
