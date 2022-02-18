#!/usr/bin/env python3
"""
Module contains function for answering questions
with a bert-uncased-tf2-qa model given a reference text.
"""


import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Answers question using bert transformer.

    Args:
        question: Question to answer.
        reference: Text to reference for finding the answer
        to the quesion

    Return: Answer to question.
    """

    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    model = hub.load(
        "https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    q_tokens = tokenizer.tokenize(question)
    r_tokens = tokenizer.tokenize(reference)
    print(len(r_tokens))

    tokens = ['[CLS]'] + q_tokens + ["[SEP]"] + \
    r_tokens + ["[SEP]"]

    word_ids = tokenizer.convert_tokens_to_ids(tokens)
    mask = [1] * len(word_ids)
    type_ids = [0] * (2 + len(q_tokens)) + [1] * \
    (len(r_tokens) + 1)

    word_ids, mask, type_ids = \
    map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (word_ids, mask, type_ids)
    )
    outputs = model([word_ids, mask, type_ids])
    start = tf.argmax(outputs[0][0][1:]) + 1
    end = tf.argmax(outputs[1][0][1:]) + 1
    answer = tokens[start: end + 1]
    answer = tokenizer.convert_tokens_to_string(answer)
    return answer
