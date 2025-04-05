import torch
import tqdm

import dataset


def test(
    model: torch.nn.Module,
    test_dataset: dataset.RavdessDataset,
    device: str,
    batch_size: int,
):
    """Testing function for Emotion Recognition model"""

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"Testing on {device}")
    model = model.to(device)

    accuracy = 0.0
    loss = 0.0

    model.eval()
    with torch.no_grad():
        for i, (waveform, emotion) in (progress := tqdm.tqdm(enumerate(test_loader))):
            waveform = waveform.to(device)
            emotion = emotion.to(device)

            prediction = model(waveform)

            loss += loss_fn(prediction, emotion).item()

            accuracy += (prediction.argmax(1) == emotion).float().mean().item()

            progress.set_postfix_str(
                f"Batch {i + 1}/{len(test_loader)}, Loss: {loss.item()}, Accuracy: {accuracy / (i + 1)}"
            )

    return loss, accuracy / len(test_loader)
