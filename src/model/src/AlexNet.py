import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

class AlexNet(Model):
    def __init__(self, 
                 num_classes,
                 input_shape,
                 hidden_layer_sizes=[32, 64],
                 kernel_sizes=[3],
                 activation="ReLU",
                 pool_sizes=[2],
                 dropout=0.5):
      
        super(AlexNet, self).__init__()

        self.conv1 = Conv2D(filters=hidden_layer_sizes[0], 
                            kernel_size=kernel_sizes[0], 
                            activation=activation,
                            input_shape=input_shape)
        self.pool1 = MaxPooling2D(pool_size=pool_sizes[0])

        self.conv2 = Conv2D(filters=hidden_layer_sizes[-1], 
                            kernel_size=kernel_sizes[-1], 
                            activation=activation)

        self.pool2 = MaxPooling2D(pool_size=pool_sizes[-1])
        self.flatten = Flatten()
        self.dropout = Dropout(dropout)
        self.fc = Dense(num_classes)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
