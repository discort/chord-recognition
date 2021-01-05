import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SequentialSampler, DataLoader

from chord_recognition.ann_utils import convert_onehot_ann
from chord_recognition.models import deep_auditory_v2
from chord_recognition.models import postprocess_HMM
from chord_recognition.utils import batch_exponential_smoothing,\
    log_filtered_spectrogram, Rescale, context_window, read_audio, one_hot


@torch.no_grad()
def forward(model, dataloader, device, num_classes, criterion=None):
    if not criterion:
        criterion = nn.Softmax(dim=1)

    result = []
    for inputs in dataloader:
        inputs = inputs.to(device=device, dtype=torch.float32)
        scores = model(inputs)
        scores = criterion(scores)
        scores = scores.squeeze(3).squeeze(2)
        result.append(scores)

    result = torch.cat(result)
    return result


class ChordRecognition:
    """
    Encapsulates the Chord Recognition Pipeline

    Args:
        window_size: FFT window size
        hop_length: number audio of frames between STFT columns
        ext_minor: label is used for a minor chord ('m' by default)
        nonchord: include or exclude non-chord class (bool)
        postprocessing: whether or not to use postprocessing.
            Availables options are `None` or `hmm`.
            if `hmm` - Hidden Markov Model postprocessing is used
    """

    def __init__(self,
                 window_size=8192,
                 hop_length=4096,
                 ext_minor=None,
                 nonchord=True,
                 postprocessing='hmm'):
        self.postprocessing = postprocessing
        self.window_size = window_size
        self.hop_length = hop_length
        self.ext_minor = ext_minor
        self.num_classes = 25
        self.nonchord = nonchord
        self.model = deep_auditory_v2(pretrained=True, model_name='deep_auditory_v2_exp4_3.pth')
        self.model.eval()  # set model to evaluation mode

    def extract_features(self, audio_waveform, sr):
        """
        Extract features from a given audio_waveform

        Args:
            audio_waveform: Audio time series (np.ndarray [shape=(n,))
            sr: Sampling rate (int)

        Returns:
            features matrix, (np.ndarray [shape=(num_features, N))
        """
        spec = log_filtered_spectrogram(
            audio_waveform=audio_waveform,
            sr=sr,
            window_size=self.window_size,
            hop_length=self.hop_length,
            fmin=65,
            fmax=2100,
            num_bands=24
        )
        return spec

    def predict_labels(self, features):
        """
        Predict chord labels from a feature-matrix

        Args:
            features - (np.ndarray [shape=(num_features, N))

        Returns:
            logits - np.ndarray [shape=(N, num_classes)]
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        dataloader = self._prepare_dataloader(features)
        ann_matrix = forward(self.model, dataloader, device, self.num_classes)
        ann_matrix = ann_matrix.data.numpy()
        return ann_matrix

    def process(self, audio_waveform, sr):
        """
        Args:
            audio_waveform: Audio time series (np.ndarray [shape=(n,))
            sr: Sampling rate (int)

        Returns:
            chords in [(start, end, label), ...] format
        """
        features = self.extract_features(audio_waveform, sr)
        features = self.preprocess(features)
        logits = self.predict_labels(features)
        labels = self.postprocess(logits)
        chords = self.decode_chords(labels, sr)
        return chords

    def preprocess(self, features):
        """
        Preprocess features matrix

        Args:
            features - (np.ndarray [shape=(num_features, N))

        Returns:
            features matrix, (np.ndarray [shape=(num_features, N))
        """
        return features

    def postprocess(self, logits):
        """
        Postprocess logits to get one-hot repr of labels

        Args:
            logits np.ndarray [shape=(N, num_classes)]

        Returns:
            one-hot representation of labels, np.array [shape=(N, num_classes)]
        """
        if not self.postprocessing:
            preds = np.argmax(logits, 1)
            labels = one_hot(preds, self.num_classes)
        elif self.postprocessing == 'hmm':
            labels = postprocess_HMM(logits.T, p=0.22).T
        else:
            raise ValueError(f"Invalid param: {postprocessing} for postprocessing")
        return labels

    def decode_chords(self, labels, sr):
        """
        Decodes labels from one-hot to human-readable format

        Args:
            labels - one-hot np.array [shape=(N, num_classes)]

        Returns:
            chords in [(start, end, label), ...] format
        """
        chords = convert_onehot_ann(
            ann_matrix=labels,
            hop_length=self.hop_length,
            sr=sr,
            ext_minor=self.ext_minor,
            nonchord=self.nonchord)
        return chords

    def _prepare_dataloader(self,
                            spectrogram,
                            context_size=7,
                            transform=Rescale(),
                            batch_size=32):
        frames = context_window(spectrogram, context_size)
        frames = np.asarray([transform(f).reshape(1, *f.shape) for f in frames])

        sampler = BatchSampler(
            sampler=SequentialSampler(frames),
            batch_size=batch_size,
            drop_last=False)
        dataloader = DataLoader(
            dataset=frames,
            sampler=sampler,
            batch_size=None)
        return dataloader


def estimate_chords(audio_path,
                    duration=None,
                    ext_minor=None,
                    nonchord=True):
    """
    Estimate chords from given audio

    Args:
        audio_path: path to an audio file
        duration: only load up to this much audio (in seconds)
        ext_minor: suffix-label for minor chords ('m' by default)
        nonchord: include or exclude non-chord class (bool)

    Returns:
        chords in [(start, end, label), ...] format
    """
    audio_waveform, sr = read_audio(
        path=audio_path,
        duration=duration)
    recognition = ChordRecognition(
        ext_minor=ext_minor,
        nonchord=nonchord)
    chords = recognition.process(
        audio_waveform=audio_waveform,
        sr=sr)
    return chords
