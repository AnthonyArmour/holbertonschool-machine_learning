#!/usr/bin/env python3
"""
   Transfer Learning script
"""


import tensorflow.keras as K


def preprocess_data(X, Y):
    """Processes input data for inceptionResNet"""
    X = K.applications.inception_resnet_v2.preprocess_input(X)
    return X, K.utils.to_categorical(Y, 10)


(x_train, y_train), (x_valid, y_valid) = K.datasets.cifar10.load_data()

x_train, y_train = preprocess_data(x_train, y_train)
x_valid, y_valid = preprocess_data(x_valid, y_valid)

_, w, h, c = x_train.shape

ResNet = K.applications.InceptionResNetV2(
    include_top=False, weights='imagenet', input_shape=(299, 299, 3)
    )
Adam = K.optimizers.Adam()

# Preprocessing layers
inputs = K.Input(shape=(32, 32, 3))
input = K.layers.Lambda(
    lambda image: K.backend.resize_images(
        image, 299/32, 299/32, data_format='channels_last'
        )
        )(inputs)

# Add base-model
layer = ResNet(input, training=False)
layer = K.layers.GlobalAveragePooling2D()(layer)

# Stack new 3 layer network
layer = K.layers.Dense(512)(layer)
layer = K.layers.BatchNormalization()(layer)
layer = K.layers.Activation('relu')(layer)
layer = K.layers.Dropout(0.3)(layer)

layer = K.layers.Dense(512)(layer)
layer = K.layers.BatchNormalization()(layer)
layer = K.layers.Activation('relu')(layer)
layer = K.layers.Dropout(0.3)(layer)

out = K.layers.Dense(10, activation='softmax')(layer)

model = K.Model(inputs, out)

# Set all InceptionResNetV2 layers to non-trainable
ResNet.trainable = False
for layer in ResNet.layers:
    layer.trainable = False


# compile the model should be done *after* setting layers to non-trainable
model.compile(
    optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy']
    )

# Train Model
model.fit(
    x_train, y_train, batch_size=300, shuffle=True,
    epochs=2, validation_data=(x_valid, y_valid), verbose=True
    )

# Unfreeze elected layers from IRNV2
for layer in ResNet.layers[:498]:
    layer.trainable = False
for layer in ResNet.layers[498:]:
    layer.trainable = True

print("\nTraining bottom couple hundred layers\n")

Adam = K.optimizers.Adam(1e-5)
model.compile(optimizer=Adam, loss="categorical_crossentropy", metrics=['acc'])
history = model.fit(
    x_train, y_train, validation_data=(x_valid, y_valid),
    batch_size=300, epochs=5, verbose=True)
