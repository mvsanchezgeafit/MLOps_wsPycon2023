from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf

class Classifier(Model):
    def __init__(self, input_shape, hidden_layer_1, hidden_layer_2, num_classes):
        super(Classifier, self).__init__()
        
        # Guardamos los parámetros para el get_config
        self.input_shape = input_shape
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.num_classes = num_classes

        # Definimos las capas con keras y activación ReLU
        self.dense1 = Dense(hidden_layer_1, activation='relu')
        self.dense2 = Dense(hidden_layer_2, activation='relu')
        self.dense3 = Dense(num_classes)  

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def get_config(self):
        """Devuelve la configuración del modelo para guardarlo"""
        config = super(Classifier, self).get_config()
        config.update({
            'input_shape': self.input_shape,
            'hidden_layer_1': self.hidden_layer_1,
            'hidden_layer_2': self.hidden_layer_2,
            'num_classes': self.num_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruye el modelo a partir de la configuración"""
        return cls(
            input_shape=config['input_shape'],
            hidden_layer_1=config['hidden_layer_1'],
            hidden_layer_2=config['hidden_layer_2'],
            num_classes=config['num_classes']
        )
