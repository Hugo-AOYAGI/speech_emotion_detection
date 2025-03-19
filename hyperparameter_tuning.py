import torch
import torchaudio
from torch.utils.data import DataLoader
import optuna
import tqdm
import traceback
import matplotlib.pyplot as plt
from dataset import RavdessDataset
from models.audio_features import AudioFeaturesModel
from train import train_ser, TrainingParameters

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 6, 8, 12, 16, 32])
    kernel_size_conv1 = trial.suggest_int("kernel_size_conv1", 3, 7)
    kernel_size_conv2 = trial.suggest_int("kernel_size_conv2", 3, 7)
    kernel_size_conv3 = trial.suggest_int("kernel_size_conv3", 3, 7)

    audio_dataset = RavdessDataset()
    train_dataset, valid_dataset = audio_dataset.random_split()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AudioFeaturesModel()
    model.conv1 = torch.nn.Conv1d(in_channels=model.n_mfcc + 2, out_channels=32, kernel_size=kernel_size_conv1, padding=4, stride=2)
    model.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size_conv2, padding=4, stride=2)
    model.conv3 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size_conv3, padding=4, stride=2)

    model = model.to(device)

    params = TrainingParameters(
        epochs=10,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device
    )

    model = train_ser(train_dataset, valid_dataset, model, params)

    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for waveform, emotion in tqdm.tqdm(valid_loader, desc="Evaluating", leave=False):
            waveform = waveform.to(device)
            emotion = emotion.to(device)

            outputs = model(waveform)
            loss = loss_fn(outputs, emotion)
            total_loss += loss.item()

            correct += (outputs.argmax(dim=1) == emotion).sum().item()
            total_samples += emotion.size(0)

    avg_loss = total_loss / len(valid_loader)
    accuracy = correct / total_samples

    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Best parameters:", study.best_params)
    print("Best validation accuracy:", study.best_value)
