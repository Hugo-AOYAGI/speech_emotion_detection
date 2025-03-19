import torch
import torch.nn as nn
from transformers import WhisperProcessor, WhisperModel


class EmotionRecognitionWhisper(torch.nn.Module):
    def __init__(self, n_classes: int = 8, whisper_model_size: str = "small"):
        super(EmotionRecognitionWhisper, self).__init__()

        self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{whisper_model_size}")
        self.model = WhisperModel.from_pretrained(f"openai/whisper-{whisper_model_size}")

        self.hidden_size = self.model.config.d_model

        self.linear = torch.nn.Linear(self.hidden_size, 512)
        self.classifier = torch.nn.Linear(512, n_classes)

    def forward(self, waveform, sampling_rate=16000):
        waveform = waveform.cpu()

        features_list = []
        for single_waveform in waveform:
            inputs = self.processor(single_waveform, sampling_rate=sampling_rate, return_tensors="pt")
            input_features = inputs.input_features.to(self.model.device)

            # Extract features using only the encoder part
            encoder_outputs = self.model.encoder(input_features).last_hidden_state
            pooled_output = encoder_outputs.mean(dim=1)

            features_list.append(pooled_output.squeeze(0))

        features = torch.stack(features_list, dim=0)
        features = torch.nn.functional.relu(self.linear(features))
        output = self.classifier(features)

        return output


if __name__ == "__main__":
    whisper_model_size = "small"
    sampling_rate = 16000

    model = EmotionRecognitionWhisper(n_classes=8, whisper_model_size=whisper_model_size)
    waveform = torch.randn(2, sampling_rate * 5)  # batch of 2 audio samples, each 5 seconds

    prediction = model(waveform)
    print(prediction.shape)
