#!/usr/bin/env python3
"""
Module contains functions for answering questions in a loop
with a bert-uncased-tf2-qa model given a reference text.
"""


import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference, tokenizer, model):
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

def answer_loop(reference):
    """
    Qustion answer loop with answers generated by
    bert transformer model.

    Args:
        reference: Text for bert model to reference for
        retrieving answers to questions.
    """

    leave = ["exit", "quit", "goodbye", "bye"]
    confused = "Sorry, I do not understand your question."

    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    model = hub.load(
        "https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    while True:
        print("Q: ", end="")
        q = input()
        if q.lower() in leave:
            print("A: Goodbye")
            break

        A = question_answer(q, reference, tokenizer, model)

        if A == "":
            print("A:", confused)
        else:
            print("A:", A)
