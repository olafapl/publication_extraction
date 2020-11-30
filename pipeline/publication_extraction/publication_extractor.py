from typing import List


class PublicationExtractor:
    """Base class for implementations of the publication string extraction module."""

    def extract_publications(self, text: str, source: str) -> List[str]:
        """Extracts publication strings from a web page.

        Args:
            text (str): Text (body text) of web page to extract publication strings from.
            source (str): Source (HTML) of web page to extract publications from.

        Returns:
            List[str]: Extracted publication strings.
        """
        pass
