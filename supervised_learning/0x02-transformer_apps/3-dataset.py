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

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_train = self.data_train.filter(self.check_len)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(self.check_len)
        self.data_valid = self.data_valid.padded_batch(batch_size)

    def check_len(self, a, b):
        """Checks for size == max_len"""
        c = tf.logical_and(tf.size(a) <= self.max_len,
                           tf.size(b) <= self.max_len)
        return c

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

    def encode(self, pt, en):
        """
        Encodes a translation into tokens.

        Args:
            pt: tf.Tensor - Portuguese sentence.
            en: tf.Tensor - Corresponding English sentence.

        Return: pt_tokens, en_tokens
            pt_tokens: np.ndarray - Portuguese tokens
            en_tokens: np.ndarray - English tokens
        """

        pt_tokens = [self.tokenizer_pt.vocab_size] + \
            self.tokenizer_pt.encode(pt.numpy().decode('utf-8')) + \
                    [self.tokenizer_pt.vocab_size+1]
        en_tokens = [self.tokenizer_en.vocab_size] + \
            self.tokenizer_en.encode(en.numpy().decode('utf-8')) + \
                    [self.tokenizer_en.vocab_size+1]
        return (pt_tokens, en_tokens)

    def tf_encode(self, pt, en):
        """
        Acts as a tensorflow wrapper for the encode instance method.

        Args:
            pt: tf.Tensor - Portuguese sentence.
            en: tf.Tensor - Corresponding English sentence.

        Return: pt_tokens, en_tokens
            pt_tokens: tf.Tensor - Portuguese tokens
            en_tokens: tf.Tensor - English tokens
        """
        a, b = tf.py_function(func=self.encode, inp=[pt, en],
                              Tout=(tf.int64, tf.int64))
        a.set_shape([None])
        b.set_shape([None])
        return a, b
