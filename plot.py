"""
Utilities for plotting training model results.
"""

import json

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_logs(train: str, test: str, epochs: int | None = None):
    """Plot on the same graph the training and validation accuracies"""

    with open(train, "r") as f:
        train_logs = [json.loads(line) for line in f.readlines()]

    with open(test, "r") as f:
        test_logs = [json.loads(line) for line in f.readlines()]

    assert len(train_logs) == len(
        test_logs
    ), "Train and test logs must have the same length"

    if epochs is not None:
        train_logs = train_logs[:epochs]
        test_logs = test_logs[:epochs]

    train_accuracy = [log["accuracy"] for log in train_logs]
    test_accuracy = [log["accuracy"] for log in test_logs]

    plt.title("Training and Testing Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(train_accuracy, label="Train Accuracy")
    plt.plot(test_accuracy, label="Test Accuracy")
    plt.legend()
    plt.grid()

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()
