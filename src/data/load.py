import os
#Se cambian las librerias de pythorch por librerias como numpy, sklearn, tensorflow
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
import argparse
import wandb

#prueba
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load(train_size=.8):
    """
    # Load the data
    """
      
    # Se carga dataset con keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    # Se hace partición del dataset con sklearn
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=train_size, random_state=42)

    training_set = (x_train, y_train)
    validation_set = (x_val, y_val)
    test_set = (x_test, y_test)
    datasets = [training_set, validation_set, test_set]
    return datasets

def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(
        project="Proyecto",
        name=f"Load Raw Data ExecId-{args.IdExecution}", job_type="load-data") as run:
        
        datasets = load()  # Separate code for loading the datasets
        names = ["training", "validation", "test"]

        # 🏺 create our Artifact
        raw_data = wandb.Artifact(
            "mnist-raw:latest", type="dataset",
            description="raw MNIST dataset, split into train/val/test",
            metadata={"source": "keras.datasets.mnist", #Se utiliza keras.datasets.mnist
                      "sizes": [len(dataset[0]) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 Store a new file in the artifact, and write something into its contents.
            with raw_data.new_file(name + ".npz", mode="wb") as file: # Se guardan los datos en formato comprimido de numpy
                x, y = data
                np.savez_compressed(file, x=x, y=y)

        # ✍️ Save the artifact to W&B.
        run.log_artifact(raw_data)

# testing
load_and_log()
