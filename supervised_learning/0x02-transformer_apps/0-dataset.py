#!/usr/bin/env python3
"""
Start of Dataset class.
"""


import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset():
    """
    Loads and preps a dataset for machine translation.
    """

    def __init__(self, batch_size, max_len):
        """
        Class constructor.

        Args:
            bacth_size: Batch size for training/validation.
            max_len: Maximum number of tokens allowed per sentence.

        Attributes:
            data_train: ted_hrlr_translate/pt_to_en tf.data.Dataset
            train split, loaded as_supervided.
            data_valid: ted_hrlr_translate/pt_to_en tf.data.Dataset
            validation split, loaded as_supervided.
            tokenizer_pt: Portuguese tokenizer created from the training set.
            tokenizer_en: English tokenizer created from the training set.
            max_len: Maximum number of tokens allowed per sentence.
        """
        self.max_len = max_len
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train', as_supervised=True)
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset.

        Args:
            data: tf.data.Dataset - a tuple (pt, en).
                pt: tf.Tensor - The Portuguese sentence.
                en: tf.Tensor - The corresponding English sentence.

        Return: tokenizer_pt, tokenizer_en
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """

        f = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus

        tokenizer_en = f((en.numpy() for _, en in data),
                         target_vocab_size=2**15)
        tokenizer_pt = f((pt.numpy() for pt, _ in data),
                         target_vocab_size=2**13)
        return tokenizer_pt, tokenizer_en
