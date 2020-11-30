from requests_html import HTMLSession
from typing import List

from .publication_extraction import PublicationExtractor
from .metadata_extraction import MetadataExtractor, Publication


class Pipeline:
    """Class for the publication metadata extraction pipeline."""

    def __init__(
        self,
        publication_extractor: PublicationExtractor,
        metadata_extractor: MetadataExtractor,
    ):
        self.html_session = HTMLSession()
        self.publication_extractor = publication_extractor
        self.metadata_extractor = metadata_extractor

    def extract_publications_from_url(self, url: str) -> List[Publication]:
        response = self.html_session.get(url).html
        source = response.html
        text = response.find("body", first=True).text
        return self.extract_publications(text, source)

    def extract_publications(self, text: str, source: str) -> List[Publication]:
        publication_strings = self.publication_extractor.extract_publications(
            text, source
        )
        return self.metadata_extractor.parse_publications(publication_strings)
