"""AST model for audio classification.

https://huggingface.co/docs/transformers/main/en/model_doc/audio-spectrogram-transformer#transformers.ASTForAudioClassification.forward.example
"""

import torch
from torch import Tensor, nn
from transformers import ASTForAudioClassification, AutoFeatureExtractor


class AST(nn.Module):
    def __init__(self, n_classes=8) -> None:
        super().__init__()

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

        self.linear = nn.Linear(527, n_classes)

    def forward(self, waveform: Tensor):
        device = waveform.device

        embeddings = torch.zeros((len(waveform), 1024, 128))
        waveform = waveform.cpu()
        for i, inputs in enumerate(waveform):
            embeddings[i] = self.feature_extractor(
                inputs, return_tensors="pt", sampling_rate=16_000
            )["input_values"]

        embeddings = embeddings.to(device)
        result = self.model(input_values=embeddings)

        return self.linear(result["logits"])
