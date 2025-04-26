from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf

class Classifier(Model):
    def __init__(self, input_shape, hidden_layer_1, hidden_layer_2, num_classes):
        super(Classifier, self).__init__()

        # Definimos las capas con keras y activación ReLu
        self.dense1 = Dense(hidden_layer_1, activation='relu')
        self.dense2 = Dense(hidden_layer_2, activation='relu')
        self.dense3 = Dense(num_classes)  

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
