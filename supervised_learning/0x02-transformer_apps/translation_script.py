#!/usr/bin/env python3
"""translation script"""


import tensorflow as tf
train_transformer = __import__('5-train').train_transformer
Dataset = __import__('3-dataset').Dataset
Translator = __import__('translator').Translator


tf.compat.v1.set_random_seed(0)
# parameters from paper (6, 512, 8, 2048, 64, 40, 20+)
transformer = train_transformer(4, 128, 8, 512, 32, 40, 20)
data = Dataset(32, 40)
translator = Translator(data, transformer)


portuguese = [
              "ola, meu nome e antonio",
              "Adoro aprendizado de máquina, porque torna o mundo um lugar melhor",
              "Eu também amo tacos, porque os tacos também tornam o mundo um lugar melhor",
              "A tradução automática é incrível!"
]
english = [
           "hello, my name is Anthony",
           "I love machine learning, because it makes the world a better place",
           "I also love tacos, because tacos also make the world a better place",
           "Machine translation is awesome!"
]


for pt, true_translation in zip(portuguese, english):
    translator.translate(pt)
    print("Real translation: ", true_translation, end="\n\n")
