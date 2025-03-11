import torch
from transformers import AutoProcessor, AutoModel


class EmotionRecognitionHubert(torch.nn.Module):

    def __init__(self, n_classes: int = 8):
        super(EmotionRecognitionHubert, self).__init__()

        self.model = AutoModel.from_pretrained("facebook/hubert-base-ls960")

        self.linear = torch.nn.Linear(self.model.config.hidden_size, 512)
        self.classifier = torch.nn.Linear(512, n_classes)

    def forward(self, waveform):
        features = self.model(waveform).last_hidden_state[:, 0, :]
        output = self.linear(features)
        output = self.classifier(output)
        return output


if __name__ == "__main__":
    model = EmotionRecognitionHubert()

    waveform = torch.randn(1, 145000)

    prediction = model(waveform)

    print(prediction)
