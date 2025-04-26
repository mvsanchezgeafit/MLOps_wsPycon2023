

# Import the model class from the main file
from src.Classifier import Classifier
import tensorflow as tf

import os
import argparse
import wandb

#hola

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

# Check if the directory "./model" exists
if not os.path.exists("./model"):
    # If it doesn't exist, create it
    os.makedirs("./model")

# Data parameters testing
# Parámetros del modelo
input_shape = 784
num_classes = 10

model_config = {
    "input_shape": input_shape,
    "hidden_layer_1": 32,
    "hidden_layer_2": 64,
    "num_classes": num_classes
}

model = Classifier(**model_config)
model.build(input_shape=(None, input_shape))  # Necesario para poder guardar en formato .h5
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def build_model_and_log(config, model, model_name="MLP", model_description="Simple MLP"):
    with wandb.init(project="Proyecto",
                    name=f"initialize Model ExecId-{args.IdExecution}",
                    job_type="initialize-model",
                    config=config) as run:

        # Crear un artifact de tipo "model"
        model_artifact = wandb.Artifact(
            model_name,
            type="model",
            description=model_description,
            metadata=dict(config)
        )

        filename = f"initialized_model_{model_name}.h5"
        model_path = os.path.join("model", filename)
        model.save(model_path)
        
        model_artifact.add_file(model_path)
        wandb.save(model_path)
        run.log_artifact(model_artifact)

# Ejecutar función
build_model_and_log(model_config, model, model_name="linear", model_description="Simple Linear Classifier")
