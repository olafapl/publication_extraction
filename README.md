# Publication Metadata Extraction Pipeline

The publication extraction pipeline is used to extract publication metadata from web pages. It
consists of two modules:

- A `PublicationExtractor` responsible for extracting the publication strings located within web
  pages. Everything related to this module is located in `/publication_extraction`.
- A `MetadataExtractor` responsible for extracting various metadata fields located within publication
  strings. Everything related to this module is located in `/metadata_extraction`

The pipeline itself is represented by the `Pipeline` class.

## Prerequisites

- Some models assume that pre-trained GloVe word embeddings are located in `data/glove/`.
- Some models assume that the HomePub dataset is located in `data/homepub-2500/`.
- Some models assume that the UMass Citation Field Extraction Dataset is located in `data/umass/`.

The `glove.sh`, `homepub.sh`, and `umass.sh` scripts can be used to download these.
