#!/usr/bin/env python3
"""
Module contains function and classes for
performing semantic search.
"""


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os


def semantic_search(corpus_path, sentence):
    """Function that performs semantic search on a corpus of documents.
    
    Args:
        corpus_path (str): The path to the corpus of reference documents on
            which to perform semantic search.
        sentence (str): The sentence from which to perform semantic search.
    Return:
        doc: The reference text of the document most similar to sentence.
    """
    model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = hub.load(model_url)

    docs = []
    docs.append(sentence)

    for file in os.listdir(corpus_path):
        if file.endswith('.md'):
            with open(corpus_path + '/' + file, encoding="utf-8") as f:
                docs.append(f.read())

    embedded = model(docs)

    sim = np.inner(embedded, embedded)[0, 1:]
    
    best = np.argmax(sim)

    return docs[1 + best]
