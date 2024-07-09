# Bangla News Processor

This Python package `BanglaNewsProcessor` is designed to preprocess Bangla news articles, extract named entities, calculate relevance scores for queries, and provide entity-based summaries. It utilizes various NLP techniques tailored for Bengali text processing.

## Features

- **Text Preprocessing**: Clean and tokenize Bangla text, remove stopwords and normalize Unicode.
- **Named Entity Recognition (NER)**: Identifies and tags Bengali named entities using `BengaliNER`.
- **Query Expansion**: Expands queries by tagging entities to improve search accuracy.
- **Relevance Scoring**: Calculates relevance scores between queries and news articles based on shared entities.
- **Entity-based Summarization**: Generates summaries based on relevant entities within articles.

## Dependencies

- `pandas` for data manipulation
- `numpy` for numerical operations
- `re` for regular expressions
- `pickle` for object serialization
- `bnlp` for Bengali text processing utilities (`BengaliCorpus`, `CleanText`, `BasicTokenizer`, `BengaliNER`)
- `bangla_stemmer` for Bengali stemming

## Installation

Install the necessary packages using pip:

```bash
pip install pandas numpy bnlp bangla-stemmer
