#!/usr/bin/env python3
"""
Module contains classes and functions
for training a transformer.
"""


import tensorflow as tf
import tensorflow.keras as K
import numpy as np


create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer
Dataset = __import__('3-dataset').Dataset


class Schedule(K.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate scheduler"""

    def __init__(self, dim, warmup=4000):
        super(Schedule, self).__init__()
        """Class constructor."""

        self.dim = tf.cast(dim, tf.float32)
        self.warmup = warmup

    def __call__(self, step):
        """Returns learning rate at step."""
        a = tf.math.rsqrt(step)
        b = step * (self.warmup ** -1.5)
        return tf.math.rsqrt(self.dim) * tf.minimum(a, b)


class Loss():

    def __init__(self):
        """Class constructor."""
        self.obj = K.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    def function(self, true, pred):
        """
        Custom loss function.

        Args:
            true: True values.
            pred: predicted values.

        Return: loss
        """
        mask = tf.math.logical_not(tf.math.equal(true, 0))
        loss = self.obj(true, pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        return tf.reduce_sum(loss)/tf.reduce_sum(mask)


SIG = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64)]


class Train():
    """Class for training transformer model."""

    def __init__(self, dim, transformer):
        """Class constructor."""
        self.lr = Schedule(dim)
        self.opt = K.optimizers.Adam(
            self.lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )
        self.loss_obj = Loss()
        self.train_loss = K.metrics.Mean(name='train_loss')
        self.accuracy = K.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )
        self.transformer = transformer

    @tf.function(input_signature=SIG)
    def train_step(self, inp, target):
        """
        Trains a step for the transformer.

        Args:
            inp: Input language sequence.
            target: Target language sequence.
        """

        tar_inp = target[:, :-1]
        targ = target[:, 1:]

        masks = create_masks(inp, tar_inp)
        enc_mask, comb_mask, dec_mask = masks

        with tf.GradientTape() as tape:
            preds, _ = self.transformer(inp, tar_inp, True, enc_mask,
                                        comb_mask, dec_mask)
            loss = self.loss_obj.function(targ, preds)

        grad = tape.gradient(loss, self.transformer.trainable_variables)
        self.opt.apply_gradients(
            zip(grad, self.transformer.trainable_variables))

        self.train_loss(loss)
        self.accuracy(targ, preds)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Creates and trains a transformer model for machine translation
    of Portuguese to English.

    Args:
        N: Blocks in the encoder and decoder
        dm: Dimensionality of the model
        h: Number of heads
        hidden: Hidden units in the fully connected layers
        max_len: Maximum number of tokens per sequence
        batch_size: Batch size for training
        epochs: Number of epochs to train for

    Return: The trained model.
    """

    data = Dataset(batch_size, max_len)

    inp_vocab_size = data.tokenizer_pt.vocab_size + 2
    tar_vocab_size = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(N, dm, h, hidden,
                              inp_vocab_size, tar_vocab_size,
                              pe_in=inp_vocab_size,
                              pe_target=tar_vocab_size)

    train = Train(dm, transformer)

    ckpt_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(transformer=train.transformer,
                               optimizer=train.opt)

    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=5)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    for epoch in range(epochs):

        train.train_loss.reset_states()
        train.accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(data.data_train):
            train.train_step(inp, tar)

            if batch % 50 == 0:
                ls, acc = train.train_loss.result(), train.accuracy.result()
                print('Epoch, {} batch {}: loss {} accuracy {}'.format(
                    epoch + 1, batch, ls, acc))

        if (epoch + 1) % 5 == 0:
            path = manager.save()
            print('Saving checkpoint at {}'.format(epoch+1, path))

        ls, acc = train.train_loss.result(), train.accuracy.result()
        print('Epoch {}: loss {} accuracy {}'.format(epoch + 1, ls, acc))
