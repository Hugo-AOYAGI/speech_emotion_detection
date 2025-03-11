from models.hubert import *
from models.wav2vec import *
from train import *
from dataset import *
import click

import torch


@click.group()
def main():
    pass


@main.command()
@click.option("--epochs", default=10, help="Number of epochs to train for.")
@click.option("--learning_rate", default=1e-5, help="Learning rate for the optimizer.")
@click.option("--batch_size", default=6, help="Batch size for training.")
@click.option("--device", default="cuda", help="Device to run the training on.")
@click.option(
    "--model_type",
    default="wav2vec",
    type=click.Choice(["wav2vec", "hubert"]),
    help="Type of model to train.",
)
@click.option("--model_path", default=None, help="Path to the model.")
def train(
    model_type: str,
    model_path: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: str,
):

    # Initialize SER model based on model_type
    match model_type:
        case "wav2vec":
            model = EmotionRecognitionWav2Vec()
        case "hubert":
            model = EmotionRecognitionHubert()
        case _:
            raise ValueError("Model type not supported.")

    # Load model from model_path if provided
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path))
        except FileNotFoundError as e:
            print("Model file not found.")
            return e

    audio_dataset = RavdessDataset()
    train_dataset, valid_dataset = audio_dataset.random_split(0.8, 0.2)

    params = TrainingParameters(
        epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, device=device
    )

    train_ser(train_dataset, valid_dataset, model, params)

    torch.save(model.state_dict(), "emotion_recognition_model.pth")


@main.command()
@click.option("--batch_size", default=6, help="Batch size for testing.")
@click.option("--device", default="cuda", help="Device to run the testing on.")
@click.option(
    "--model_type",
    default="wav2vec",
    type=click.Choice(["wav2vec", "hubert"]),
    help="Type of model to test.",
)
@click.option(
    "--model_path",
    default="checkpoints/emotion_recognition_model.pth",
    help="Path to the model.",
)
def test(model_type: str, model_path: str, batch_size: int, device: str):

    # Initialize SER model based on model_type
    match model_type:
        case "wav2vec":
            model = EmotionRecognitionWav2Vec()
        case "hubert":
            model = EmotionRecognitionHubert()
        case _:
            raise ValueError("Model type not supported.")

    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError as e:
        print("Model file not found.")
        return e

    audio_dataset = RavdessDataset()
    _, test_dataset = audio_dataset.random_split(0.8, 0.2)

    loss, accuracy = test(model, test_dataset, device, batch_size)

    print(f"Loss: {loss}, Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
