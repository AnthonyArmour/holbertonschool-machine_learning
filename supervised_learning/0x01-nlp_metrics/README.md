# NLP BLEU Score Evaluation Metric

[BLEU Score](https://en.wikipedia.org/wiki/BLEU)

## Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |

## Tasks

### [Unigram BLEU score](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-nlp_metrics/0-uni_bleu.py "Unigram BLEU score")
Calculates the unigram bleu score for a sentence.

``` python
#!/usr/bin/env python3

uni_bleu = __import__('0-uni_bleu').uni_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(uni_bleu(references, sentence))
```

```
0.6549846024623855
```

### [N-gram BLEU score](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-nlp_metrics/1-ngram_bleu.py "N-gram BLEU score")
Calculates the n-gram bleu score for a sentence.

``` python
#!/usr/bin/env python3

ngram_bleu = __import__('1-ngram_bleu').ngram_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(ngram_bleu(references, sentence, 2))
```

```
0.6140480648084865
```


### [Cumulative N-gram BLEU score](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-nlp_metrics/2-cumulative_bleu.py "Cumulative N-gram BLEU score")
Calculates the cumulative n-gram bleu score for a sentence.

``` python
#!/usr/bin/env python3

cumulative_bleu = __import__('1-cumulative_bleu').cumulative_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(cumulative_bleu(references, sentence, 4))
```

```
0.5475182535069453
```