import click
import torch
from torch.utils.data import DataLoader

from dataset import *
from models.ast import AST
from models.audio_features import *
from models.hubert import *
from models.majority_vote import MajorityVote
from models.speechbrain import SpeechBrain
from models.wav2vec import *
from models.whisper import *
from plot import plot_logs
from tests import test as test_model
from train import *


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
    type=click.Choice(
        ["wav2vec", "hubert", "audio_features", "ast", "speechbrain", "whisper"]
    ),
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
        case "audio_features":
            model = AudioFeaturesModel()
        case "ast":
            model = AST()
        case "speechbrain":
            model = SpeechBrain(device=device)
        case "whisper":
            model = EmotionRecognitionWhisper()
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
    train_dataset, valid_dataset = audio_dataset.random_split()

    params = TrainingParameters(
        epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, device=device
    )

    train_ser(train_dataset, valid_dataset, model, model_type, params)

    torch.save(model.state_dict(), f"{model_type}.pth")


@main.command()
@click.option("--batch_size", default=6, help="Batch size for testing.")
@click.option("--device", default="cuda", help="Device to run the testing on.")
@click.option(
    "--model_type",
    default="wav2vec",
    type=click.Choice(["wav2vec", "hubert", "audio_features", "ast", "speechbrain"]),
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
        case "audio_features":
            model = AudioFeaturesModel()
        case "ast":
            model = AST()
        case "speechbrain":
            model = SpeechBrain(device=device)
        case _:
            raise ValueError("Model type not supported.")

    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError as e:
        print("Model file not found.")
        return e

    audio_dataset = RavdessDataset()
    _, test_dataset = audio_dataset.random_split()

    loss, accuracy = test_model(model, test_dataset, device, batch_size)

    print(f"Loss: {loss}, Accuracy: {accuracy}")


@main.command()
@click.option(
    "--type",
    default="vote",
    type=click.Choice(["vote", "softmax"]),
    help="Type of majority vote.",
)
@click.option("--device", default="cpu", help="Device to run the testing on.")
def majority_vote(type: str, device: str):
    # Load models on the CPU
    ast = AST().eval()
    hubert = EmotionRecognitionHubert().eval()
    wav2vec = EmotionRecognitionWav2Vec().eval()

    ast.load_state_dict(torch.load("checkpoints/ast.pth"))
    hubert.load_state_dict(torch.load("checkpoints/hubert.pth"))
    wav2vec.load_state_dict(torch.load("checkpoints/wav2vec.pth"))

    ast.to(device)
    hubert.to(device)
    wav2vec.to(device)

    # Create a Majority Vote model
    majority_vote = MajorityVote([ast, hubert, wav2vec])

    audio_dataset = RavdessDataset()
    _, test_dataset = audio_dataset.random_split()

    loader = DataLoader(test_dataset, batch_size=6)

    correct = 0
    total = 0

    with torch.no_grad():
        for waveform, emotion in tqdm.tqdm(loader):
            prediction = majority_vote(waveform.to(device), method=type).cpu()
            correct += (prediction == emotion).sum().item()
            total += len(emotion)

    print(f"Accuracy: {correct / total}")


@main.command()
@click.argument("train", type=click.Path(exists=True))
@click.argument("test", type=click.Path(exists=True))
def plot(train: str, test: str):
    """Plot on the same graph the training and validation accuracies"""

    plot_logs(train, test)


if __name__ == "__main__":
    main()
