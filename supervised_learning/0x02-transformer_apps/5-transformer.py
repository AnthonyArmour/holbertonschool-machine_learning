#!/usr/bin/env python3
"""
Module contains a series of classes for building
a transformer.
"""


import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Layer, Dense
import numpy as np


def angles(pos, i, dim):
    """Executing formula for positional encoding."""
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(dim))
    return pos * angle_rates


def positional_encoding(pos, dim):
    """
    Positional encoding for relative word position.

    Args:
        pos: Position
        dim: Model dimension
    """
    angle_rads = angles(np.arange(pos)[:, np.newaxis],
                        np.arange(dim)[np.newaxis, :], dim)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_attention(q, k, v, mask):
    """
    Calculate the attention weights.

    Args:
        q: query - shape (..., seq_len_q, depth)
        k: key - shape (..., seq_len_k, depth)
        v: value - shape (..., seq_len_v, depth_v)
        mask: Float tensor -
        (..., seq_len_q, seq_len_k). Defaults to None.

    Return: output, attention_weights
    """
    qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled = qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled += (mask * -1e9)

    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, v)

    return output, weights


def point_wise_FFN(model_dim, dim2):
    """Pointwise feed-forward network."""
    ffn = K.Sequential([
        Dense(dim2, activation='relu'),
        Dense(model_dim)
    ])
    return ffn


class EncoderLayer(Layer):
    """Layer architecture for encoder layer."""

    def __init__(self, dim, heads, dim2, rate=0.1):
        super(EncoderLayer, self).__init__()
        """
        Class constructor.

        Args:
            dim: Dimensionality of the model
            heads: Number of heads
            dim2: Hidden units in the fully connected layers
            rate: Dropout rate
        """

        self.attention = MultiHead(dim, heads)
        self.ffn = point_wise_FFN(dim, dim2)

        self.norm1 = K.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = K.layers.LayerNormalization(epsilon=1e-6)

        self.drop1 = K.layers.Dropout(rate)
        self.drop2 = K.layers.Dropout(rate)

    def call(self, x, train, mask):
        """Returns output of encoder layer"""
        att_out, _ = self.attention(x, x, x, mask)
        att_out = self.drop1(att_out, training=train)
        norm = self.norm1(x + att_out)

        ffn_out = self.ffn(norm)
        ffn_out = self.drop2(ffn_out, training=train)
        out = self.norm2(norm + ffn_out)

        return out


class MultiHead(Layer):
    """Multi headed attention network"""

    def __init__(self, dim, heads):
        super(MultiHead, self).__init__()
        """
        Class constructor.

        Args:
            dim: Dimension of model output.
            heads: Number of heads.
        """

        self.heads = heads
        self.dim = dim

        assert dim % heads == 0

        self.depth = dim // heads
        self.wq = Dense(dim)
        self.wk = Dense(dim)
        self.wv = Dense(dim)
        self.dense = Dense(dim)

    def split(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """Returns output of multi headed attention network."""
        bs = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split(q, bs)
        k = self.split(k, bs)
        v = self.split(v, bs)

        attention, weights = scaled_attention(q, k, v, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, (bs, -1, self.dim))
        output = self.dense(attention)

        return output, weights


class DecoderLayer(Layer):
    """Layer architecture for decoder layer"""

    def __init__(self, dim, heads, dim2, rate=0.1):
        super(DecoderLayer, self).__init__()
        """
        Class constructor.

        Args:
            dim: Dimensionality of the model
            heads: Number of heads
            dim2: Hidden units in the fully connected layers
            rate: Dropout rate
        """

        self.att1 = MultiHead(dim, heads)
        self.att2 = MultiHead(dim, heads)

        self.ffn = point_wise_FFN(dim, dim2)

        self.norm1 = K.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = K.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = K.layers.LayerNormalization(epsilon=1e-6)

        self.drop1 = K.layers.Dropout(rate)
        self.drop2 = K.layers.Dropout(rate)
        self.drop3 = K.layers.Dropout(rate)

    def call(self, x, enc_out, train, look_ahead, padding_mask):
        """Returns output for decoder layer."""

        att1, weights1 = self.att1(x, x, x, look_ahead)
        att1 = self.drop1(att1, training=train)
        norm1 = self.norm1(att1 + x)

        att2, weights2 = self.att2(enc_out, enc_out, norm1, padding_mask)
        att2 = self.drop2(att2, training=train)
        norm2 = self.norm2(att2 + norm1)

        ffn_out = self.ffn(norm2)
        ffn_out = self.drop3(ffn_out, training=train)
        out = self.norm3(ffn_out + norm2)

        return out, weights1, weights2


class Encoder(Layer):
    """Encoder network for transformer"""

    def __init__(self, layers, dim, heads, dim2,
                 vocab_size, max_pos_encoding, rate=0.1):
        super(Encoder, self).__init__()
        """
        Class constructor.

        Args:
            layers: Number of encoder layers
            dim: Dimensionality of the model
            heads: Number of heads
            dim2: Hidden units in the fully connected layers
            rate: Dropout rate
            vocab_size: Input corpus vocab_size
            max_pos_encoding: sequence position limit
        """

        self.dim = dim
        self.layers = layers

        self.embedding = K.layers.Embedding(vocab_size, dim)
        self.pos_encoding = positional_encoding(max_pos_encoding, dim)

        self.encoder_layers = [EncoderLayer(dim, heads, dim2, rate)
                               for _ in range(layers)]

        self.drop = K.layers.Dropout(rate)

    def call(self, x, train, mask):
        """Returns output of encoder blocks"""

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.drop(x, training=train)

        for i in range(self.layers):
            x = self.encoder_layers[i](x, train, mask)

        return x


class Decoder(Layer):
    """Decoder network for transformer"""

    def __init__(self, layers, dim, heads, dim2,
                 vocab_size, max_pos_encoding, rate=0.1):
        super(Decoder, self).__init__()
        """
        Class constructor.

        Args:
            layers: Number of decoder layers
            dim: Dimensionality of the model
            heads: Number of heads
            dim2: Hidden units in the fully connected layers
            rate: Dropout rate
            vocab_size: Input corpus vocab_size
            max_pos_encoding: sequence position limit
        """

        self.dim = dim
        self.layers = layers

        self.embedding = K.layers.Embedding(vocab_size, dim)
        self.pos_encoding = positional_encoding(max_pos_encoding, dim)

        self.decoder_layers = [DecoderLayer(dim, heads, dim2, rate)
                               for _ in range(layers)]
        self.drop = K.layers.Dropout(rate)

    def call(self, x, enc_out, train, look_ahead, padding_mask):
        """Returns output of decoder"""

        seq_len = tf.shape(x)[1]
        weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.drop(x, training=train)

        for i in range(self.layers):
            tup = self.decoder_layers[i](
                x, enc_out, train, look_ahead, padding_mask)
            x, weights1, weights2 = tup
            weights["decoder_weights1_layer{}".format(i+1)] = weights1
            weights["decoder_weights2_layer{}".format(i+1)] = weights2

        return x, weights


class Transformer(K.Model):
    """Transformer for machine translation"""

    def __init__(self, layers, dim, heads, dim2, in_vocab_size,
                 target_vocab_size, pe_in, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        """
        Class constructor.

        Args:
            layers: Number of encoder layers
            dim: Dimensionality of the model
            heads: Number of heads
            dim2: Hidden units in the fully connected layers
            rate: Dropout rate
            in_vocab_size: Input corpus vocab_size
            target_vocab_size: Target corpus vocab_size
            pe_in: input sequence position limit
            pe_target: taget sequence position limit
        """

        self.encoder = Encoder(layers, dim, heads, dim2,
                               in_vocab_size, pe_in, rate)
        self.decoder = Decoder(layers, dim, heads, dim2,
                               target_vocab_size, pe_target, rate)

        self.last_layer = Dense(target_vocab_size)

    def call(self, inp, tar, train, enc_mask, look_ahead, dec_mask):
        """Returns output for transformer"""

        enc_out = self.encoder(inp, train, enc_mask)
        dec_out, att_weights = self.decoder(
            tar, enc_out, train, look_ahead, dec_mask)

        out = self.last_layer(dec_out)
        return out, att_weights
