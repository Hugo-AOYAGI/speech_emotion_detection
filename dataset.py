import os
import kagglehub
import torchaudio

import torch
import tqdm

# RADVESS dataset path taken from Kaggle
DATASET_URL = "uwrfkaggler/ravdess-emotional-speech-audio"
EMOTIONS = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fear",
    6: "disgust",
    7: "surprise",
}
SAMPLE_RATE = 16000


class RavdessDataset(torch.utils.data.Dataset):
    """Custom class for the Ravdess Audio dataset taken from Kaggle"""

    def __init__(self):
        super(RavdessDataset, self).__init__()

        self.path = kagglehub.dataset_download(DATASET_URL)
        ravdess_directory_list = os.listdir(self.path)

        self.emotions = []
        self.waveforms = []

        audio_file_paths = []

        # Loop through the directories and extract the audio files
        for dir in ravdess_directory_list:

            actor = os.listdir(os.path.join(self.path, dir))
            for file in actor:
                if ".wav" not in file:
                    continue
                part = file.split(".")[0]
                part = part.split("-")

                # The emotion information is contained in the file name itself
                self.emotions.append(int(part[2]) - 1)
                audio_file_paths.append(os.path.join(self.path, dir, file))

        for audio_file_path in tqdm.tqdm(audio_file_paths, desc="Loading Audio Files"):
            # Load the audio file using torchaudio
            waveform, sample_rate = torchaudio.load(audio_file_path, format="wav")
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=SAMPLE_RATE
            )(waveform)
            if waveform.shape[0] > 1:  # If dual-channel, convert to mono
                waveform = waveform.mean(0, keepdim=True)
            self.waveforms.append(waveform.squeeze(0))

        self.waveforms = torch.nn.utils.rnn.pad_sequence(
            self.waveforms, batch_first=True
        )

        self.sample_rate = sample_rate

    def random_split(self, train_size, valid_size):
        return torch.utils.data.random_split(self, [train_size, valid_size])

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, index):
        return self.waveforms[index], self.emotions[index]


if __name__ == "__main__":
    audio_dataset = RavdessDataset()
    print("Dataset size:", len(audio_dataset))

    print("Original sample rate:", audio_dataset.sample_rate)

    # Load the first audio file
    print("Loading the first audio file")
    waveform, emotion = audio_dataset[0]
    print("Waveform shape:", waveform.shape)
    print("Emotion:", EMOTIONS[emotion])

    print("Splitting the dataset into training and validation")
    train_dataset, valid_dataset = audio_dataset.random_split(0.8, 0.2)
    print("Train and valid dataset size:", len(train_dataset), len(valid_dataset))
