import torch
import random

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class EmotionRecognitionWav2Vec(torch.nn.Module):

    def __init__(self, n_classes: int = 8):
        super(EmotionRecognitionWav2Vec, self).__init__()

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        self.classifier = torch.nn.Linear(self.model.config.vocab_size, n_classes)

    def forward(self, waveform):
        inputs = self.processor(
            waveform, return_tensors="pt", device=self.model.device, sampling_rate=16000
        )
        inputs = inputs.input_values.view(waveform.shape[0], -1).to(self.model.device)
        features = self.model(input_values=inputs).logits[:, -1, :].squeeze(1)
        output = self.classifier(features)
        return output


if __name__ == "__main__":
    model = EmotionRecognitionWav2Vec()

    waveform = torch.randn(1, random.randint(100000, 200000))

    prediction = model(waveform)

    print(prediction)
