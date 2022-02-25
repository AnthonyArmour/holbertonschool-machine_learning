#!/usr/bin/env python3
"""
Module contains function and classes for
performing semantic search.
"""


import tensorflow as tf
import tensorflow_hub as hub
semantic_search = __import__('3-semantic_search').semantic_search
import numpy as np
import os


def get_answer(question, reference, tokenizer, model):
    """
    Answers question using bert transformer.

    Args:
        question: Question to answer.
        reference: Text to reference for finding the answer
        to the quesion
        tokenizer: BertTokenizer for tokenizing text.
        model: Bert transformer model.

    Return: Answer to question.
    """

    q_tokens = tokenizer.tokenize(question)
    r_tokens = tokenizer.tokenize(reference)

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

def question_answer(corpus_path):
    """
    QA bot for answering questions based on a corpus
    of reference documents.

    Args:
        corpus_path: Path to folder containing
        reference documents.
    """

    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    model = hub.load(
        "https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    path = "https://tfhub.dev/google/universal-sentence-encoder/4"

    leave = ["exit", "quit", "goodbye", "bye"]
    confused = "Sorry, I do not understand your question."

    while True:
        print("Q: ", end="")
        q = input()
        if q.lower() in leave:
            print("A: Goodbye")
            break

        ref = semantic_search(corpus_path, q)
        A = get_answer(q, ref, tokenizer, model)

        if A == "":
            print("A:", confused)
        else:
            print("A:", A)
