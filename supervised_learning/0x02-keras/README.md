# Introduction to Keras Project

## Tools Created:

* build_model(nx, layers, activations, lambtha, keep_prob):

> Builds a neural network with the Keras library.

* build_model(nx, layers, activations, lambtha, keep_prob):

> Builds a neural network with the Keras library.

* optimize_model(network, alpha, beta1, beta2):

> Sets up Adam optimization for a keras model with categorical crossentropy loss and accuracy metrics.

* one_hot(labels, classes=None):

> One hot encode function to be used to reshape Y_label vector.

* train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False):

> Trains a model using mini-batch gradient descent.

* train_model(network, data, labels, batch_size, epochs, validation_data=none, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False):

> Trains a model using mini-batch gradient descent.

* save_model(network, filename):

> Saves Model

* load_model(filename):

> loads model

* save_weights(network, filename, save_format='h5'):

> Saves weights of the model

* load_weights(network, filename):

> Loads weights of a model.

* save_config(network, filename):

> Saves config of model to json.

* load_config(filename):

> Loads model with specific cofiguration

* test_model(network, data, labels, verbose=True):

> Tests model on testing data.

* predict(network, data, verbose=False):

> Makes a prediction using a neural network.
