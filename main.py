from baseline import *
from train import *
from dataset import *

import torch

if __name__ == "__main__":
    model = EmotionRecognitionBaseline()
    audio_dataset = RavdessDataset()
    train_dataset, valid_dataset = audio_dataset.random_split(0.8, 0.2)

    params = TrainingParameters(
        epochs=10, learning_rate=0.00005, batch_size=6, device="cuda"
    )

    train(train_dataset, valid_dataset, model, params)

    torch.save(model.state_dict(), "emotion_recognition_model.pth")
