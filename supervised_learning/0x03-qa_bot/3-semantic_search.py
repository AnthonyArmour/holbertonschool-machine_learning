#!/usr/bin/env python3
"""
Module contains function and classes for
performing semantic search.
"""


import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np
from numpy.linalg import norm
import os
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer



class KeyWordFrequency():
    """
    Class for getting the average tf-idf score
    for sentences with respect to key words
    from a given question or sentence.
    """

    def __init__(self, corpus_path=None):
        """
        Class constructor.

        Args:
            corpus_path: Path to reference texts.

        Attributes:
            corpus_path: Path to reference texts.
            stopwords: List of common english stopwords.
        """
        self.corpus_path = corpus_path
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

    def set_vocab(self, sen):
        """
        Sets words as vocab for vectorizer.

        Args:
            sen: sentence to parse and set as vocab.
        """
        self.words = self.remove_punc(sen).lower().split()
        self.remove_stop_words()

    def remove_stop_words(self):
        """
        Removes stop words from words attribute.
        """
        for word in self.words.copy():
            if word in self.stopwords:
                self.words.pop(self.words.index(word))

    @staticmethod
    def remove_punc(sen):
        """
        Removes punctuation from sentence.

        Args:
            sen: Sentence to remove punctuations from.

        Return:
            Edited sentence.
        """
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        
        for ele in sen:
            if ele in punc:
                sen = sen.replace(ele, "")
        return sen

    @staticmethod
    def tf_idf(sentences, vocab=None):
        """
        Creates a TF-IDF embedding.
        Args:
            sentences: List of sentences to analyze.
            vocab: List of the vocabulary words to use
            for the analysis. If None, all words within
            sentences should be used.
        Return: embeddings, features
            embeddings: numpy.ndarray - (s, f) Embeddings.
                s: Number of sentences in sentences.
                f: Number of features analyzed.
            features: List of the features used for embeddings.
        """

        vec = TfidfVectorizer(use_idf=True, vocabulary=vocab, stop_words="english")
        X = vec.fit_transform(sentences)

        emb = X.toarray()
        return emb

    def keywords(self, sen_list=None):
        """
        Gets mean tf-idf score for each sentence in
        sen_list.

        Args:
            sen_list: List of sentences.

        Return:
            mean tf-idf score for each sentence in sen_list.
        """

        if sen_list is None:
            files = os.listdir(self.corpus_path)
            docs = []

            for i, file in enumerate(files):
                try:
                    with open("./"+self.corpus_path+"/"+file) as f:
                        d = f.read()
                except UnicodeDecodeError:
                    d = ""
                docs.append(d)

            kw = self.tf_idf(docs, self.words)
        else:
            kw = self.tf_idf(sen_list, self.words)

        kw = np.mean(kw, axis=1) + 0.1
        return kw


class SemanticSimilarity():
    """
    Class for getting a semantic similarity score
    for sentences with respect to a question or sentence.
    """

    def __init__(self, path):
        """
        Class constructor.

        Args:
            path: Path to tensorflow hub
            universal-sentence-encoder

        Attributes:
            model: universal-sentence-encoder model.
        """
        self.model = hub.load(path)

    def embed(self, input):
        """Returns semantic embedding from text."""
        return self.model(input)

    def semantic_dist(self, doc, sen_, mu=True):
        """
        Calculates the semantic distance between a list of
        sentences and a target sentence.

        Args:
            doc: List of sentences from reference document.
            sen_: Target sentence.
            mu: bool, if True take the mean semantic distance across all
            sentences.

        Return: Semantic distance
        """
        emb = self.embed(doc)
        if mu:
            emb = np.mean(emb, axis=0)
        text = self.embed(sen_)
        return self.dist(text, emb)

    def dist(self, emb1, emb2):
        """
        Cosine simularity calculation between two
        sets of embeddings.

        Args:
            emb1: First embedding set.
            emb2: Second embedding set.
            Embedding sets can have multiple embeddings
            along the first axis.

        Return: Cosine similarity.
        """
        nm = norm(emb1) * norm(emb2)
        return np.inner(emb1, emb2) / nm


def semantic_search(corpus_path, sentence, SS=None, KWF=None):
    """
    Finds the document with the highest semantic similarity
    and tf-idf score.

    Args:
        corpus_path: Path to reference documents.
        sentence: Target sentence or question.
        SS: SemanticSimilarity object containing
        universal-sentence-encoder model
        KWF: KeyWordFrequency object for computing
        term frequency - inverse document frequency scores

    Returns: Document text with highest relevence to the
    target sentence or question.
    """

    if SS is None:
        path = "https://tfhub.dev/google/universal-sentence-encoder/4"
        SS = SemanticSimilarity(path)
    if KWF is None:
        KWF = KeyWordFrequency(corpus_path)

    KWF.set_vocab(sentence)
    gamma = 0.9
    docs = os.listdir(corpus_path)
    semantic_dist = []

    for i, doc in enumerate(docs):
        try:
            with open("./"+corpus_path+"/"+doc) as f:
                d = f.read()
        except UnicodeDecodeError:
            semantic_dist.append(0)
            continue
        semantic_dist.append(SS.semantic_dist(d.split("."), [sentence])[0])

    semantic_dist = np.array(semantic_dist)
    keyword_freq = KWF.keywords()
    scores = semantic_dist * (gamma * keyword_freq)

    best = np.argmax(scores)

    with open("./"+corpus_path+"/"+docs[best]) as f:
        doc = f.read()

    return doc
