"""Speechbrain classification model.
"""

from speechbrain.inference.speaker import EncoderClassifier
from torch import Tensor, nn


class SpeechBrain(nn.Module):

    model: EncoderClassifier

    def __init__(self, n_classes=8, device: str = "cuda") -> None:
        super().__init__()

        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb",
            run_opts={"device": device},
        )  # type: ignore

        self.linear = nn.Linear(512, n_classes)

    def forward(self, x: Tensor):

        embeddings = self.model.encode_batch(x)
        return self.linear(embeddings).squeeze(-2)
