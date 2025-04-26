

import os
import argparse
import wandb
import numpy as np 
import pickle
##hola
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

def preprocess(x, y, normalize=True, expand_dims=True):
    x = x.astype('float32')

    if normalize:
        x = x / 255.0

    if expand_dims and len(x.shape) == 3:
        x = np.expand_dims(x, axis=-1)

    return x, y

def preprocess_and_log(steps):

    with wandb.init(project="Proyecto",name=f"Preprocess Data ExecId-{args.IdExecution}", job_type="preprocess-data") as run:    
        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="Preprocessed MNIST dataset",
            metadata=steps)
         
        # ✔️ declare which artifact we'll be using
        raw_data_artifact = run.use_artifact('mnist_raw:v0', type='dataset')

        # 📥 if need be, download the artifact
        raw_dataset = raw_data_artifact.download(root="./data/artifacts/")
        
        for split in ["training", "validation", "test"]:
            x, y = read(raw_dataset, split)
            x, y = preprocess(x, y, **steps)

            # Guardamos en formato .npz (compresión NumPy)
            filename = split + ".npz"
            filepath = os.path.join("./", filename)
            np.savez_compressed(filepath, x=x, y=y)

            # Lo añadimos al artifact de salida
            processed_data.add_file(filepath, name=filename)

        run.log_artifact(processed_data)

 


def read(data_dir, split):
    filename = split + ".npz"
    filepath = os.path.join(data_dir, filename)

    with np.load(filepath) as data:
        x = data["x"]
        y = data["y"]

    return x, y


steps = {"normalize": True,
         "expand_dims": False}

preprocess_and_log(steps)
