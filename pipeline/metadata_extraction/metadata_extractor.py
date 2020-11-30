from typing import List


class Publication:
    def __init__(
        self, author: str = None, title: str = None, venue: str = None, year: int = None
    ):
        self.author = author
        self.title = title
        self.venue = venue
        self.year = year


class MetadataExtractor:
    """Base class for implementations of the publication metadata extraction module."""

    def parse_publications(
        self,
        publication_strings: List[str],
    ) -> List[Publication]:
        """Parses and extracts metadata from publication strings.

        Args:
            publication_strings (List[str]): Publication strings.

        Returns:
            List[Publication]: Parsed publication strings.
        """
        pass
