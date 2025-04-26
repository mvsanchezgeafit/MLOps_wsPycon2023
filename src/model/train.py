
# Import 
from src.Classifier import Classifier

import os
import argparse
import wandb
import numpy as np 

import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

# Device configuration 
if tf.config.list_physical_devices('GPU'):
    device = "GPU"
else:
    device = "CPU"

print("Device:", device)

def read(data_dir, split):
    """

    """
    filename = split + ".npz"
    filepath = os.path.join(data_dir, filename)

    with np.load(filepath) as data:
        x = data["x"]
        y = data["y"]

    return x, y




def train(model, train_loader, valid_loader, config):

    optimizer_class = getattr(tf.keras.optimizers, config.optimizer)
    optimizer = optimizer_class()

    # Función de pérdida
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Métricas para monitorear
    train_loss = tf.keras.metrics.Mean()
    val_loss = tf.keras.metrics.Mean()
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  
    #model.train()
    example_ct = 0
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
                # 🔁 Entrenamiento
        for step, (x_batch, y_batch) in enumerate(train_loader):
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss_value = loss_fn(y_batch, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            train_loss.update_state(loss_value)
            example_ct += x_batch.shape[0]

            if step % config.batch_log_interval == 0:
                loss_value_scalar = loss_value.numpy()  # Convertir a tipo escalar para WandB
                print(f"Step {step}, Loss: {loss_value_scalar:.4f}")
                train_log(loss_value_scalar, example_ct, epoch)
       

        # 🧪 Validación al final de la época
        for x_batch_val, y_batch_val in valid_loader:
            val_logits = model(x_batch_val, training=False)
            v_loss = loss_fn(y_batch_val, val_logits)

            val_loss.update_state(v_loss)
            val_accuracy.update_state(y_batch_val, val_logits)

        print(f"Validation Loss: {val_loss.result():.4f}, Accuracy: {val_accuracy.result():.4f}")
        test_log(val_loss.result().numpy(), val_accuracy.result().numpy(), example_ct, epoch)

        # Reset metrics
        train_loss.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()
    
def test(model, test_loader):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for x_batch, y_batch in test_loader:
        logits = model(x_batch, training=False)
        loss_value = loss_fn(y_batch, logits)

        test_loss.update_state(loss_value)
        test_accuracy.update_state(y_batch, logits)

    return test_loss.result().numpy(), test_accuracy.result().numpy() * 100

def train_log(loss, example_ct, epoch):
    loss = float(loss)
    # where the magic happens
    wandb.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    

def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)
    # where the magic happens
    wandb.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
    print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{accuracy:.3f}")

def evaluate(model, test_loader):
    """
    ## Evaluate the trained model
    """

    loss, accuracy = test(model, test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader.dataset)

    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions

def get_hardest_k_examples(model, testing_set, k=32):
    # Convertimos el conjunto de prueba en un tf.data.Dataset si no lo está
    # Esto asume que `testing_set` es una tupla de arrays (x_test, y_test).
    x_test, y_test = testing_set
    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)

    # Función de pérdida
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    losses = []
    predictions = []

    # Deshabilitamos el entrenamiento
    model.eval()

    # Calculamos las pérdidas y predicciones para cada ejemplo
    for x_batch, y_batch in dataset:
        logits = model(x_batch, training=False)  # Predicción
        loss_value = loss_fn(y_batch, logits)   # Pérdida

        # Calculamos la predicción más probable
        pred = tf.argmax(logits, axis=1, output_type=tf.int32)

        losses.append(loss_value.numpy())  # Guardamos la pérdida
        predictions.append(pred.numpy())  # Guardamos las predicciones

    # Convertimos las pérdidas y predicciones a numpy arrays
    losses = np.array(losses)
    predictions = np.array(predictions)

    # Obtenemos los indices de las pérdidas más altas
    argsort_loss = np.argsort(losses)

    # Seleccionamos los k ejemplos más difíciles
    highest_k_losses = losses[argsort_loss[-k:]]
    hardest_k_examples = x_test[argsort_loss[-k:]]
    true_labels = y_test[argsort_loss[-k:]]
    predicted_labels = predictions[argsort_loss[-k:]]

    return highest_k_losses, hardest_k_examples, true_labels, predicted_labels





def train_and_log(config, experiment_id='99'):
    with wandb.init(
        project="Proyecto", 
        name=f"Train Model ExecId-{args.IdExecution} ExperimentId-{experiment_id}", 
        job_type="train-model", config=config) as run:

        config = wandb.config
        data = run.use_artifact('mnist-preprocess:latest')
        data_dir = data.download()

        x_train, y_train = read(data_dir, "training")
        x_valid, y_valid = read(data_dir, "validation")
        x_train = x_train.reshape((-1, 784))
        x_valid = x_valid.reshape((-1, 784))

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(config.batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(config.batch_size)

        # ⚠️ Crea el modelo directamente en lugar de cargarlo desde archivo
        model = Classifier(input_shape=784, hidden_layer_1=128, hidden_layer_2=64, num_classes=10)
        model.build((None, 784))  # necesario para poder guardarlo después

        model.compile(
            optimizer=config.optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

        model.fit(train_dataset, validation_data=val_dataset, epochs=config.epochs)

        # Guardamos el modelo en el nuevo formato .keras
        model.save("trained_model.keras")

        model_artifact = wandb.Artifact(
            "trained-model", type="model",
            description="Trained NN model",
            metadata=dict(config)
        )
        model_artifact.add_file("trained_model.keras")
        run.log_artifact(model_artifact)

    return model    
def evaluate_and_log(experiment_id='99', config=None):
    with wandb.init(
        project="Proyecto",
        name=f"Eval Model ExecId-{args.IdExecution} ExperimentId-{experiment_id}",
        job_type="eval-model",
        config=config) as run:

        data = run.use_artifact('mnist-preprocess:latest')
        data_dir = data.download()
        x_test, y_test = read(data_dir, "test")
        x_test = x_test.reshape((-1, 784))
        testing_set = (x_test, y_test)
        test_dataset = tf.data.Dataset.from_tensor_slices(testing_set).batch(128)

        model_artifact = run.use_artifact("trained-model:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "trained_model.keras")

        model = tf.keras.models.load_model(model_path)

        loss, accuracy = model.evaluate(test_dataset)
        run.summary.update({"loss": loss, "accuracy": accuracy})

        highest_losses, hardest_examples, true_labels, preds = get_hardest_k_examples(model, testing_set, k=32)

        wandb.log({
            "high-loss-examples": [
                wandb.Image(hard_example, caption=f"{int(pred)}," + str(int(label))) 
                for hard_example, pred, label in zip(hardest_examples, preds, true_labels)
            ]
        })

# Para ejecutar las evaluaciones con diferentes configuraciones de épocas
epochs = [50, 100, 200]
for id, epoch in enumerate(epochs):
    train_config = {
        "batch_size": 128,
        "epochs": epoch,
        "batch_log_interval": 25,
        "optimizer": "Adam"
    }
    model = train_and_log(train_config, id)  # Entrenamos el modelo
    evaluate_and_log(id)  # Evaluamos el modelo

"""    
train_config = {"batch_size": 128,
                "epochs": 5,
                "batch_log_interval": 25,
                "optimizer": "Adam"}

model = train_and_log(train_config)
evaluate_and_log()
"""

