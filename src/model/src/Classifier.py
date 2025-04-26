from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class Classifier(Model):
    def __init__(self, input_shape, hidden_layer_1, hidden_layer_2, num_classes, **kwargs):
        super(Classifier, self).__init__(**kwargs)

        # Guardar parámetros del modelo
        self.input_shape_config = input_shape  # renombramos para evitar conflicto con built-in
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.num_classes = num_classes

        # Capas
        self.dense1 = Dense(hidden_layer_1, activation='relu')
        self.dense2 = Dense(hidden_layer_2, activation='relu')
        self.dense3 = Dense(num_classes)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_config,
            "hidden_layer_1": self.hidden_layer_1,
            "hidden_layer_2": self.hidden_layer_2,
            "num_classes": self.num_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            input_shape=config["input_shape"],
            hidden_layer_1=config["hidden_layer_1"],
            hidden_layer_2=config["hidden_layer_2"],
            num_classes=config["num_classes"]
        )
