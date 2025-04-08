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

    # Store (correct, total) for each class. Used to compute accuracy later
    accuracy_per_class = {i: (0, 0) for i in range(8)}

    model.eval()
    with torch.no_grad():
        for i, (waveform, emotion) in (progress := tqdm.tqdm(enumerate(test_loader))):
            waveform = waveform.to(device)
            emotion = emotion.to(device)

            prediction = model(waveform)

            loss += loss_fn(prediction, emotion).item()

            correct = (prediction.argmax(1) == emotion).float()
            accuracy += correct.mean().item()

            for real, success in zip(emotion, correct):
                # Update the total count for the class
                index = real.item()
                accuracy_per_class[index] = (
                    accuracy_per_class[index][0] + success.item(),
                    accuracy_per_class[index][1] + 1,
                )

            progress.set_postfix_str(
                f"Batch {i + 1}/{len(test_loader)}, Loss: {loss}, Accuracy: {accuracy / (i + 1)}"
            )

    # Compute accuracy per class
    for i in range(8):
        correct, total = accuracy_per_class[i]
        if total > 0:
            accuracy_per_class[i] = correct / total  # type: ignore
        else:
            accuracy_per_class[i] = 0.0  # type: ignore

    print("Accuracy per class:")
    for i in range(8):
        print(accuracy_per_class[i], end=",")

    print()
    print()

    return loss, accuracy / len(test_loader)
