import tensorflow.keras as k
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dense, BatchNormalization, Conv2D, Flatten, add
from tensorflow.keras.optimizers import Adam


class Policy():

    def __init__(self, action_space, lr=0.0001, beta=0.00005, convL=2, fcL=(2, 256), filters=[16, 16],
                 activation="tanh", kernel_size=[(5, 5), (3, 3)]):
        self.action_space = action_space
        self.lr = lr
        self.convL = convL
        self.fcL = fcL
        self.filters = filters
        self.beta = beta
        self.activation = activation
        self.kernel_size = kernel_size


    def ConvResNet(self):

        inp = Input(shape=(7, 6, 1))
        initializer = k.initializers.HeNormal()
        conv_add = BatchNormalization()(inp)

        for layer in range(self.convL):
            conv = Conv2D(
                filters=self.filters[layer], kernel_size=self.kernel_size[layer], padding="same", use_bias=True, bias_initializer='ones'
                )(conv_add)
            conv = BatchNormalization()(conv)
            convA = Activation(self.activation)(conv)

            conv = Conv2D(
                filters=self.filters[layer], kernel_size=self.kernel_size[layer], padding="same", use_bias=True, bias_initializer='ones'
                )(convA)
            conv = BatchNormalization()(conv)
            convB = Activation(self.activation)(conv)

            conv_add = add([convA, convB])


        conv = Conv2D(filters=64, kernel_size=(3, 3), padding="same", use_bias=True, bias_initializer='ones')(conv_add)
        conv = BatchNormalization()(conv)
        conv = Activation(self.activation)(conv)

        fc_add = Flatten()(conv)

        for layer in range(int(self.fcL[0]/2)):
            fc = Dense(units=self.fcL[1], kernel_initializer=initializer, use_bias=True, bias_initializer='ones')(fc_add)
            fc = BatchNormalization()(fc)
            fcA = Activation(self.activation)(fc)

            fc = Dense(units=self.fcL[1], kernel_initializer=initializer, use_bias=True, bias_initializer='ones')(fcA)
            fc = BatchNormalization()(fc)
            fcB = Activation(self.activation)(fc)

            fc_add = add([fcA, fcB])
        
        return fc_add, initializer, inp



    def get_policy_ResConvNet(self, paths=None):

        fc, init, inp = self.ConvResNet()

        action = Dense(units=self.action_space, activation="softmax", kernel_initializer=init, kernel_regularizer='l2')(fc)

        Policy = Model(inp, action)
        Policy.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr))

        return Policy
