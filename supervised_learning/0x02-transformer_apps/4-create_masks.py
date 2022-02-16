#!/usr/bin/env python3
"""
Module contains functions for making masks.
"""


import tensorflow as tf


def padding_mask(seq):
    """Creates padded mask"""
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def look_ahead_mask(size):
    """Creates look ahead mask"""
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask[tf.newaxis, ...]


def create_masks(inputs, target):
    """
    Creates all masks for training/validation.

    Args:
        inputs: tf.Tensor - (batch_size, seq_len_in) - input sentence
        target: tf.Tensor - (batch_size, seq_len_out) - target sentence

    Return:
    encoder_mask: Mask to be applied in the encoder
    combined_mask: Maximum between a lookaheadmask and the decoder
    target padding mask.
    decoder_mask: Used in the 2nd attention block in the decoder.
    """
    enc_mask = padding_mask(inputs)
    dec_mask = padding_mask(inputs)
    look_ahead = look_ahead_mask(tf.shape(target)[1])
    tar_mask = padding_mask(target)
    combined_mask = tf.maximum(tar_mask, look_ahead)

    return enc_mask, combined_mask, dec_mask
