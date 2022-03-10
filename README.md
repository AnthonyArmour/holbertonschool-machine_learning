# Machine Learning Repository for School
This repository contains over 30 projects related to machine learning! For learning purposes, most of these implementations are done from scratch using numpy, although, for some of the projects there is a healthy amount of tensorflow, keras, pytorch, scipy, pandas,  and/or matplotlib. So far, this repo covers a very wide space of different machine learning algorithms. I'd be honored if you explore them. I'll list and link some of my favorite project folders in this readme.




# My Favorite Projects :blush:


[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)

# [0. Object Detection Project](https://github.com/AnthonyArmour/holbertonschool-machine_learning/tree/master/supervised_learning/0x00-object_detection)
Object detection is a computer technology related to computer vision and image processing that deals with detecting instances of semantic objects of a certain class (such as humans, buildings, or cars) in digital images and videos. Well-researched domains of object detection include face detection and pedestrian detection. Object detection has applications in many areas of computer vision, including image retrieval and video surveillance.


## Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| tensorflow         | ^2.6.0  |
| keras              | ^2.6.0  |
| cv2                | ^4.1.0  |

## Model
Add to ./data folder
[yolo](https://intranet-projects-files.s3.amazonaws.com/holbertonschool-ml/yolo.h5)

## Tasks:
In this project I use the yolo v3 algorithm to perform object detection. There are multiple files building on the same class because of the structure of the assignment provided by Holberton school. The entire Yolo class can be found in 7-yolo.py which is linked below. The class methods are documented if you would like to know the inner workings.

### [Yolo Class](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/7-yolo.py "Yolo")

``` python
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('7-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('./data/yolo.h5', './data/coco_classes.txt', 0.6, 0.5, anchors)
    predictions, image_paths = yolo.predict('./data/yolo')
```
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/images/yolo-1.png)
```
Press the s button to save image:
```
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/images/yolo-2.png)
```
Press the s button to save image:
```
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/images/yolo-3.png)
```
Press the s button to save image:
```
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/images/yolo-4.png)
```
Press the s button to save image:
```
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/images/yolo-5.png)
```
Press the s button to save image:
```
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/images/yolo-6.png)
```
Press the s button to save image:
```

:small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond:

---

[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)

# [1. Transformer For Machine Translation](https://github.com/AnthonyArmour/holbertonschool-machine_learning/tree/master/supervised_learning/0x02-transformer_apps)


## Dependencies
| Library/Framework              | Version |
| ------------------------------ | ------- |
| Python                         | ^3.7.3  |
| numpy                          | ^1.19.5 |
| matplotlib                     | ^3.4.3  |
| tensorflow                     | ^2.6.0  |
| keras                          | ^2.6.0  |
| tensorflow-datasets            | ^4.5.2  |

## Tasks

### [Class Dataset](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-transformer_apps/3-dataset.py "Class Dataset")
Encodes a translation into tokens and sets up a data pipeline for the transformer model.

### [Class Transformer](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-transformer_apps/5-transformer.py "Class Transformer")
Series of classes to build transformer for machine translation.

### [Training Function](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-transformer_apps/5-train.py "Training Function")
``` python
#!/usr/bin/env python3
import tensorflow as tf
train_transformer = __import__('5-train').train_transformer

tf.compat.v1.set_random_seed(0)
transformer = train_transformer(4, 128, 8, 512, 32, 40, 20)
```

```
Epoch 1, batch 0: loss 10.26855754852295 accuracy 0.0
Epoch 1, batch 50: loss 10.23129940032959 accuracy 0.0009087905054911971

...

Epoch 1, batch 600: loss 7.164522647857666 accuracy 0.06743457913398743
Epoch 1, batch 650: loss 7.076988220214844 accuracy 0.07054812461137772
Epoch 1: loss 7.038494110107422 accuracy 0.07192815840244293
Epoch 2, batch 0: loss 5.177524089813232 accuracy 0.1298387050628662
Epoch 2, batch 50: loss 5.189461708068848 accuracy 0.14023463428020477

...

Epoch 2, batch 600: loss 4.870367527008057 accuracy 0.15659810602664948
Epoch 2, batch 650: loss 4.858142375946045 accuracy 0.15731287002563477
Epoch 2: loss 4.852652549743652 accuracy 0.15768977999687195

...

Epoch 20 batch 550 Loss 1.1597 Accuracy 0.3445
Epoch 20 batch 600 Loss 1.1653 Accuracy 0.3442
Epoch 20 batch 650 Loss 1.1696 Accuracy 0.3438
```

### [Translator](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-transformer_apps/translator.py "Translator")
Class for translating Portuguese to english via the transformer model.
Here's a script and results trained on 50 epochs. Much better performance can be achieved with hyperparameter tuning and more training. You'll find some of the translations are pretty good.

``` python
#!/usr/bin/env python3
import tensorflow as tf
train_transformer = __import__('5-train').train_transformer
translator = __import__(translator.py)

tf.compat.v1.set_random_seed(0)
transformer, data = train_transformer(4, 128, 8, 512, 32, 40, 50, ret_data=True)
translator = Translator(data, transformer)


# Some sentences that I know get good results

sentences = [
             "este é um problema que temos que resolver.",
             "os meus vizinhos ouviram sobre esta ideia.",
             "vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.",
             "este é o primeiro livro que eu fiz."
    ]

true = [
        "this is a problem we have to solve .",
        "and my neighboring homes heard about this idea .",
        "so i 'll just share with you some stories very quickly of some magical things that have happened .",
        "this is the first book i've ever done."
]

for sen, t in zip(sentences, true):
    translator.translate(sen)
    print("Real Translation: ", t, end="\n\n")


print("\n\n\n\n\n------------------------------------------\n\n\n\n\n")
print("From Test Set:\n")

test_set = tfds.load('ted_hrlr_translate/pt_to_en', split='test', as_supervised=True)

for pt, true_translation in test_set.take(32):
    translator.translate(pt.numpy().decode('utf-8'))
    print("Real translation: ", true_translation.numpy().decode('utf-8'), end="\n\n")

```

```
Input: este é um problema que temos que resolver.
Prediction: this is a problem that we have to solve .
Real Translation:  this is a problem we have to solve .

Input: os meus vizinhos ouviram sobre esta ideia.
Prediction: my neighbors heard about this idea in the united states .
Real Translation:  and my neighboring homes heard about this idea .

Input: vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.
Prediction: so i 'm going to share with you some very quickly stories that happened to be an entire magic .
Real Translation:  so i 'll just share with you some stories very quickly of some magical things that have happened .

Input: este é o primeiro livro que eu fiz.
Prediction: this is the first book i did .
Real Translation:  this is the first book i've ever done.
```

<details>
<summary>Results from test set!</summary>

```
From Test Set:

Input: depois , podem fazer-se e testar-se previsões .
Prediction: then they can do it and test them forecast .
Real translation:  then , predictions can be made and tested .

Input: forçou a parar múltiplos laboratórios que ofereciam testes brca .
Prediction: it forced to stop multiple laboratories , they offer brand-warranry .
Real translation:  it had forced multiple labs that were offering brca testing to stop .

Input: as formigas são um exemplo clássico ; as operárias trabalham para as rainhas e vice-versa .
Prediction: re-tech is a classic ; opec donors work for ques and vice versions .
Real translation:  ants are a classic example ; workers work for queens and queens work for workers .

Input: uma em cada cem crianças no mundo nascem com uma doença cardíaca .
Prediction: one in every hundred kids in the world are born with a heart disease .
Real translation:  one of every hundred children born worldwide has some kind of heart disease .

Input: neste momento da sua vida , ela está a sofrer de sida no seu expoente máximo e tinha pneumonia .
Prediction: at this moment of life , she 's suffering from aids expose in its full and i had five-scale neutral .
Real translation:  at this point in her life , she 's suffering with full-blown aids and had pneumonia .

Input: onde estão as redes económicas ?
Prediction: where are the economic networks ?
Real translation:  where are economic networks ?

Input: ( aplausos )
Prediction: ( applause )
Real translation:  ( applause )

Input: eu usei os contentores de transporte , e também os alunos ajudaram-nos a fazer toda a mobília dos edifícios , para torná-los confortáveis​​ , dentro do orçamento do governo , mas também com a mesma a área da casa , mas muito mais confortável .
Prediction: but i had really powerful transportation , and these students helped us do all the way to make them feel all the same building .
Real translation:  i used the shipping container and also the students helped us to make all the building furniture to make them comfortable , within the budget of the government but also the area of the house is exactly the same , but much more comfortable .

Input: e , no entanto , a ironia é que a única maneira de podermos fazer qualquer coisa nova é dar um passo nessa direção .
Prediction: and yet , the irony , though , is the only way we can do anything new thing is take into that direction .
Real translation:  and yet , the irony is , the only way we can ever do anything new is to step into that space .

Input: a luz nunca desaparece .
Prediction: light never disappear .
Real translation:  the light never goes out .

Input: `` agora , `` '' tweets '' '' , quem está a `` '' tweetar '' '' ? ''
Prediction: `` now tweet , '' '' who is tweet , '' to tweet , '' now ? '' ''
Real translation:  now , tweets , who 's tweeting ?

Input: no egito , por exemplo , 91 % das mulheres que vivem hoje no egito foram mutiladas sexualmente dessa forma .
Prediction: in egypt , for example , 91 percent of women who live today in egypt today were mutually just from this way .
Real translation:  in egypt , for instance , 91 percent of all the females that live in egypt today have been sexually mutilated in that way .

Input: por outro lado , os bebés de 15 meses ficavam a olhar para ela durante muito tempo caso ela agisse como se preferisse os brócolos , como se não percebessem a situação .
Prediction: on the other side , 15 months would take look at it for a very long time and she was willing to see a broccolic dog , like they did n't notice .
Real translation:  on the other hand , 15 month-olds would stare at her for a long time if she acted as if she liked the broccoli , like they could n't figure this out .

Input: naquele momento , percebi quanta energia negativa é precisa para conservar aquele ódio dentro de nós .
Prediction: at that moment , i realized how much energy is needed to conservate that hate us in us .
Real translation:  in that instant , i realized how much negative energy it takes to hold that hatred inside of you .

Input: e a discussão é : o que é que isso interessa .
Prediction: and the argument is what it matters .
Real translation:  and the discussion is , who cares ? right ?

Input: se escolhermos um lugar e formos cuidadosos , as coisas estarão sempre lá quando as procurarmos .
Prediction: if you choose a place and you can get careful , things will always be there when you look there .
Real translation:  if you designate a spot and you 're scrupulous about it , your things will always be there when you look for them .

Input: é um museu muito popular agora , e criei um grande monumento para o governo .
Prediction: it 's a very popular museum now , and i set up a large monument to the government .
Real translation:  it 's a very popular museum now , and i created a big monument for the government .

Input: é completamente irrelevante .
Prediction: it 's completely irrele .
Real translation:  it 's completely irrelevant .

Input: todos defenderam que a sua técnica era a melhor , mas nenhum deles tinha a certeza disso e admitiram-no .
Prediction: they all advocate for their technique was better , but none of them was sure of them about it , and i admitted it .
Real translation:  `` they all argued that , `` '' my technique is the best , '' '' but none of them actually knew , and they admitted that . ''

Input: a partir daquele momento , comecei a pensar .
Prediction: from that moment , i started to think .
Real translation:  at that moment , i started thinking .

Input: mt : portanto , aqui temos a maré baixa e aqui a maré alta e no centro temos a lua .
Prediction: mt : so here we have the sea-down-down , and here is high center and on the moon .
Real translation:  mt : so over here is low tide , and over here is high tide , and in the middle is the moon .

Input: então , este jogo é muito simples .
Prediction: so this game is pretty simple .
Real translation:  beau lotto : so , this game is very simple .

Input: então , propus a reconstrução . angariei , recolhi fundos .
Prediction: so i proposed to rebuilding . i raised fundamentally , i collected funds .
Real translation:  so i proposed to rebuild . i raised — did fundraising .

Input: o que nós - betty rapacholi , minha aluna , e eu - fizemos foi dar aos bebés dois pratos de comida : um prato com brócolos crus e um com bolachas deliciosas em forma de peixinho .
Prediction: what we do — jeff atmosque , and i was making carava — on the two dig babies with colonies : a rubber and a rubber ball .
Real translation:  what we did — betty rapacholi , who was one of my students , and i — was actually to give the babies two bowls of food : one bowl of raw broccoli and one bowl of delicious goldfish crackers .

Input: é algo que nos acontece sem o nosso consentimento .
Prediction: it 's something that happens without our consent .
Real translation:  it 's something that happens to us without our consent .

Input: ardemos de paixão .
Prediction: we are burning passion .
Real translation:  we burn with passion .

Input: `` a mutilação genital é horrível , e desconhecida pelas mulheres americanas . mas , nalguns países , em muitos países , quando uma menina nasce , muito cedo na sua vida , os seus genitais são completamente removidos por um chamado `` '' cortador '' '' que , com uma lâmina de navalha , sem recurso à esterilização , corta as partes exteriores dos genitais femininos . ''
Prediction: manso , so , so , inorganic parts of the american women , but , in some countries at very young countries , when they 're born very born .
Real translation:  genital mutilation is horrible and not known by american women , but in some countries , many countries , when a child is born that 's a girl , very soon in her life , her genitals are completely cut away by a so-called cutter who has a razor blade and , in a non-sterilized way , they remove the exterior parts of a woman 's genitalia .

Input: isto significa 20 % do orçamento , do orçamento relativo a cuidados de saúde do país .
Prediction: this means 20 percent of budget budget budget to health care care .
Real translation:  that 's 20 percent of the budget , of the healthcare budget of the country .

Input: conheci-o num evento 46664 .
Prediction: i know it in a 4646 event .
Real translation:  i met him at a 46664 event .

Input: deixem-me mostrar-vos o que quero dizer .
Prediction: let me show you what i mean .
Real translation:  let me show you what i mean .

Input: acho que este é o problema .
Prediction: i think this is the problem .
Real translation:  i think this is a problem .

Input: mt : oh , 365 , o número de dias num ano , o número de dias entre cada aniversário .
Prediction: mt : oh , 36 , number 5 , the number of days ago , the number of days between every birthday .
Real translation:  mt : oh , 365 , the number of days in a year , the number of days between each birthday .
```
</details>


:small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond:

---



