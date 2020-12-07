from typing import List


class Publication:
    def __init__(
        self,
        authors: List[str] = None,
        title: str = None,
        year: str = None,
        journal: str = None,
    ):
        self.authors = authors
        self.title = title
        self.year = year
        self.journal = journal

    def __str__(self):
        return (
            f"Publication(authors={self.authors}, title={self.title}, "
            f"year={self.year}, journal={self.journal})"
        )


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
