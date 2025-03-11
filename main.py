from baseline import *
from train import *
from dataset import *
import click

import torch


@click.group()
def main():
    pass


@click.command()
@click.option("--epochs", default=10, help="Number of epochs to train for.")
@click.option(
    "--learning_rate", default=0.00005, help="Learning rate for the optimizer."
)
@click.option("--batch_size", default=6, help="Batch size for training.")
@click.option("--device", default="cuda", help="Device to run the training on.")
def train(epochs, learning_rate, batch_size, device):
    model = EmotionRecognitionBaseline()
    audio_dataset = RavdessDataset()
    train_dataset, valid_dataset = audio_dataset.random_split(0.8, 0.2)

    params = TrainingParameters(
        epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, device=device
    )

    train(train_dataset, valid_dataset, model, params)

    torch.save(model.state_dict(), "emotion_recognition_model.pth")


@click.command()
@click.option("--batch_size", default=6, help="Batch size for testing.")
@click.option("--device", default="cuda", help="Device to run the testing on.")
def test(batch_size, device):
    model = EmotionRecognitionBaseline()
    model.load_state_dict(torch.load("emotion_recognition_model.pth"))

    audio_dataset = RavdessDataset()
    _, test_dataset = audio_dataset.random_split(0.8, 0.2)

    loss, accuracy = test(model, test_dataset, device, batch_size)

    print(f"Loss: {loss}, Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
