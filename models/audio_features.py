import torchaudio
import torch

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset import *


class AudioFeaturesModel(torch.nn.Module):
    def __init__(
        self,
        n_emotions: int = 8,
        window_length: int = 400,
        hop_length: int = 160,
        n_mfcc: int = 30,
        n_mels: int = 48,
    ):
        super(AudioFeaturesModel, self).__init__()

        self.n_emotions = n_emotions

        # Audio features parameters
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

        self.mfcc_model = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.window_length,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
                "center": False,
            },
        )

        # Input size is n_mfcc + 2 (pitch and rms)
        self.conv1 = torch.nn.Conv1d(
            in_channels=self.n_mfcc + 2,
            out_channels=32,
            kernel_size=3,
            padding=4,
            stride=2,
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=4,
            stride=2,
        )
        self.conv3 = torch.nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=4,
            stride=2,
        )

        self.pooling = torch.nn.AdaptiveAvgPool1d(1)

        self.classifier = torch.nn.Linear(128, self.n_emotions)

    def mfcc(self, waveform: torch.tensor):
        return self.mfcc_model(waveform)

    def pitch(self, waveform: torch.tensor):
        return torchaudio.functional.detect_pitch_frequency(
            waveform,
            sample_rate=SAMPLE_RATE,
        )

    def rms(self, waveform: torch.tensor):
        frames = waveform.unfold(
            dimension=-1, size=self.window_length, step=self.hop_length
        )
        rms_frames = torch.sqrt(torch.mean(frames**2, dim=-1))
        return rms_frames

    def forward(self, x: torch.tensor):
        # Extract audio features
        mfcc = self.mfcc(x)
        pitch = self.pitch(x).unsqueeze(-2)
        pitch = torch.nn.functional.pad(pitch, (mfcc.shape[-1] - pitch.shape[-1], 0))
        rms = self.rms(x).unsqueeze(-2)

        # Concatenate audio features
        x = torch.cat([mfcc, pitch, rms], dim=-2)

        # Pass through the model
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))

        x = self.pooling(x)
        x = x.squeeze(-1)

        return self.classifier(x)


if __name__ == "__main__":
    # Load sample audio file
    audio_dataset = RavdessDataset(minimal=True)
    print("Dataset size:", len(audio_dataset))

    wf = audio_dataset[:2][0]
    print("Waveform shape:", wf.shape)

    model = AudioFeaturesModel()

    out = model(wf)
    print("Output shape:", out.shape)
