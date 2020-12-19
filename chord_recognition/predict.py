import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SequentialSampler, DataLoader
import torch.nn.functional as F

from chord_recognition.dataset import context_window
from chord_recognition.ann_utils import compute_annotation
from chord_recognition.utils import exponential_smoothing,\
    log_filtered_spectrogram, preprocess_spectrogram
from .cnn import deep_auditory_v2


model = deep_auditory_v2(pretrained=True)
model.eval()  # set model to evaluation mode


def predict_annotations(spectrogram, model, device, batch_size=32, context_size=7, num_classes=25):
    frames = context_window(spectrogram, context_size)
    frames = [preprocess_spectrogram(f) for f in frames]
    frames = np.asarray([f.reshape(1, *f.shape) for f in frames])

    sampler = BatchSampler(
        sampler=SequentialSampler(frames),
        batch_size=batch_size,
        drop_last=False)
    dataloader = DataLoader(
        dataset=frames,
        sampler=sampler,
        batch_size=None)
    result = forward(model, dataloader, device, num_classes)
    return result.data.numpy()


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
        #scores = batch_exponential_smoothing(scores, 0.1)
        result.append(scores)

    result = torch.cat(result)
    preds = torch.argmax(result, 1)
    result = F.one_hot(preds, num_classes)
    return result


def batch_exponential_smoothing(x, alpha):
    batsches, _ = x.shape
    result = []
    for i in range(batsches):
        x_smooth = exponential_smoothing(x[i, :].numpy(), alpha)
        result.append(torch.from_numpy(x_smooth))
    return torch.stack(result)


def annotate_audio(audio_waveform, Fs, window_size=8192, hop_length=4096,
                   ext_minor=None, nonchord=True):
    """Calculate the annotation of specified audio file

    - calculates spectrogram
    - applies filterbanks
    - logarithmise the filtered magnitudes to compress the value range
    - compute annotation matrix
    - apply exponential smoothing
    - convert annotation matrix to basic ann representation

    Args:
        audio_waveform: Audio time series (np.ndarray [shape=(n,))
        Fs: Sampling rate (int)
        ext_minor: label is used for a minor chord ('m' by default)
        nonchord: include or exclude non-chord class (bool)

    Returns:
        annotation: annotated file in format [(start, end, label), ...]
    """
    # chromagram = compute_chromagram(
    #     audio_waveform=audio_waveform,
    #     Fs=Fs,
    #     window_size=window_size,
    #     hop_length=hop_length)
    spec = log_filtered_spectrogram(
        audio_waveform=audio_waveform,
        sr=Fs,
        window_size=window_size,
        hop_length=hop_length,
        fmin=65, fmax=2100, num_bands=24
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ann_matrix = predict_annotations(spec, model, device, batch_size=8)  # N x num_classes
    annotations = compute_annotation(
        ann_matrix, hop_length, Fs, ext_minor=ext_minor, nonchord=nonchord)
    return annotations
