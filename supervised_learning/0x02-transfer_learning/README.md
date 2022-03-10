[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)

# Transfer Learning Introduction Project

Check out this blog post where I talk about this project in more depth.
[blog](https://www.linkedin.com/pulse/summary-my-transfer-learning-project-using-inception-resnet-armour/)

## Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| tensorflow         | ^2.6.0  |
| keras              | ^2.6.0  |

### Transfer script using [inception resnet v2](https://keras.io/api/applications/inceptionresnetv2/)

```
./0-transfer.py
Training bottom couple hundred layers
saved to ./cifar10.h5
```

``` python
#!/usr/bin/env python3

import tensorflow.keras as K
preprocess_data = __import__('0-transfer').preprocess_data

# to fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase 

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10.h5')
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)
```

```
10000/10000 [==============================] - 159s 16ms/sample - loss: 0.2329 - acc: 0.9235
```