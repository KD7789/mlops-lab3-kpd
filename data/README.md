# Data

`raw/` holds the three files from the Sentiment Labelled Sentences dataset, included unchanged:

- `amazon_cells_labelled.txt`
- `imdb_labelled.txt`
- `yelp_labelled.txt`

Each line is a review sentence, a tab, and a label: 1 = positive, 0 = negative.

- Source: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
- Citation: Kotzias, D. (2015). Sentiment Labelled Sentences [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C57604
- License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

`stopwords.txt` is scikit-learn's English stop word list with negation words (no, not, never, etc.) removed, so negative phrases keep their meaning.
