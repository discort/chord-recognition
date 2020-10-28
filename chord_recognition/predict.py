import os.path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SequentialSampler
import torch.nn.functional as F

from chord_recognition.dataset import ContextIterator
from chord_recognition.utils import compute_chromagram, compute_annotation
from .cnn import model

CURR_DIR = os.path.dirname(__file__)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.load_state_dict(
    torch.load(os.path.join(CURR_DIR, 'models/etd_best_model.pt'), map_location=device))
model.eval()


@torch.no_grad()
def predict_annotations(spectrogram, model, device, batch_size=32, context_size=7, num_classes=25):
    criterion = nn.Softmax(dim=1)
    container = ContextIterator(spectrogram, context_size)
    frames = np.asarray([f.reshape(1, *f.shape) for f in container])

    sampler = BatchSampler(SequentialSampler(frames), batch_size=batch_size, drop_last=False)
    result = []
    for idx in sampler:
        inputs = torch.from_numpy(frames[idx]).to(device=device, dtype=torch.float32)
        scores = model(inputs)
        scores = criterion(scores)
        scores = scores.squeeze(3).squeeze(2)
        result.append(scores)

    result = torch.cat(result)
    preds = torch.argmax(result, 1)
    result = F.one_hot(preds, num_classes).t_()
    return result.data.numpy()


def annotate_audio(audio_waveform, Fs, window_size=8192, hop_length=4096, nonchord=True):
    """Calculate the annotation of specified audio file

    Args:
        audio_waveform: Audio time series (np.ndarray [shape=(n,))
        Fs: Sampling rate (int)
        nonchord: include or exclude non-chord class (bool)

    Returns:
        annotation: annotated file in format [(start, end, label), ...]
    """
    # calculate spectrogram
    # apply filterbanks
    # compute annotation matrix
    # apply moving average smoothing
    # convert annotation matrix to basic ann representation
    chromagram = compute_chromagram(
        audio_waveform=audio_waveform,
        Fs=Fs,
        window_size=window_size,
        hop_length=hop_length)
    ann_matrix = predict_annotations(chromagram, model, device, batch_size=8)
    # apply median smoothing to ann_matrix
    annotations = compute_annotation(ann_matrix, hop_length, Fs, nonchord=nonchord)
    return annotations
