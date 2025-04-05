"""Speechbrain classification model.
"""

from speechbrain.inference.speaker import EncoderClassifier
from torch import Tensor, nn


class SpeechBrain(nn.Module):

    model: EncoderClassifier

    def __init__(self, n_classes=8, device: str = "cuda") -> None:
        super().__init__()

        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )  # type: ignore

        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: Tensor):

        embeddings = self.model.encode_batch(x)
        return self.head(embeddings).squeeze(-2)
