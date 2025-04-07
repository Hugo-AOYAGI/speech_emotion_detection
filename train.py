import json
import traceback
from dataclasses import dataclass

import torch
import tqdm

import dataset


@dataclass
class TrainingParameters:
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def write_log(filename: str, data: dict):
    """Write logs to a jsonl file"""

    with open(filename, "a") as f:
        f.write(json.dumps(data) + "\n")


def train_ser(
    train_dataset: dataset.RavdessDataset,
    valid_dataset: dataset.RavdessDataset,
    model: torch.nn.Module,
    model_name: str,
    params: TrainingParameters,
):
    """Training function for Emotion Recognition model"""

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=0.01
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.batch_size, shuffle=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=params.batch_size, shuffle=True
    )

    print(f"Training on {params.device}")
    model = model.to(params.device)

    best_validation_accuracy = 0.0
    best_model = None

    try:
        for epoch in (progress := tqdm.tqdm(range(params.epochs))):
            training_loss = 0.0
            training_accuracy = 0.0
            validation_loss = 0.0
            validation_accuracy = 0.0

            model.train()

            for i, (waveform, emotion) in enumerate(train_loader):

                waveform = waveform.to(params.device)
                emotion = emotion.to(params.device)

                prediction = model(waveform)

                loss = loss_fn(prediction, emotion)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                training_accuracy += (
                    (prediction.argmax(1) == emotion).float().mean().item()
                )

                progress.set_postfix_str(
                    f"Batch {i + 1}/{len(train_loader)}, Loss: {loss.item()}"
                )

            write_log(
                f"{model_name}-train.jsonl",
                {
                    "loss": training_loss / len(train_loader),
                    "accuracy": training_accuracy / len(train_loader),
                },
            )

            with torch.no_grad():
                model.eval()
                for i, (waveform, emotion) in enumerate(valid_loader):
                    progress.set_postfix_str(f"Batch {i + 1}/{len(train_loader)}")

                    waveform = waveform.to(params.device)
                    emotion = emotion.to(params.device)

                    prediction = model(waveform)
                    loss = loss_fn(prediction, emotion)

                    validation_loss += loss.item()
                    validation_accuracy += (
                        (prediction.argmax(1) == emotion).float().mean()
                    ).item()

                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_model = model.state_dict().copy()

            write_log(
                f"{model_name}-test.jsonl",
                {
                    "loss": validation_loss / len(valid_loader),
                    "accuracy": validation_accuracy / len(valid_loader),
                },
            )

            print(
                f"Epoch {epoch + 1}/{params.epochs}\n",
                f"Training Loss: {training_loss / len(train_loader)}\n",
                f"Validation Loss: {validation_loss / len(valid_loader)}\n",
                f"Validation Accuracy: {validation_accuracy / len(valid_loader)}\n",
            )

    # Save the model if training is interrupted
    except KeyboardInterrupt:
        print("Training interrupted")
        return best_model
    except Exception as e:
        print(traceback.format_exc())

    return model
