"""Torch dataset over the mixer: snippets are rendered on the fly."""

import numpy as np
import torch
from mixer import MixConfig, iter_tracks, render_snippet
from torch.utils.data import Dataset

# Slakh2100's 13 inst_class values (as they appear in its metadata.yaml files,
# including the odd "Strings (continued)" group) plus MUSDB18's stem names.
CLASSES = [
    "Bass", "Brass", "Chromatic Percussion", "Drums", "Guitar", "Organ", "Piano",
    "Pipe", "Reed", "Strings", "Strings (continued)", "Synth Lead", "Synth Pad",
    "vocals", "drums", "bass", "other",
]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}


def class_vector(classes) -> np.ndarray:
    v = np.zeros(len(CLASSES), dtype=np.float32)
    for c in classes:
        v[CLASS_INDEX[c]] = 1.0
    return v


class SnippetDataset(Dataset):
    """`length` snippets drawn from the given tracks.

    Item `i` uses track `i % n_tracks` and seed `seed_base + i + epoch * length`
    (set `epoch` via `set_epoch` for training; leave at 0 for fixed validation
    sets)."""

    def __init__(self, tracks, config: MixConfig, length: int, seed_base: int = 0):
        self.tracks = list(tracks)
        self.config = config
        self.length = length
        self.seed_base = seed_base
        self.epoch = 0

    @classmethod
    def from_dataset(cls, dataset, split, config, length, seed_base=0):
        return cls(iter_tracks(dataset.name, dataset.path, split), config, length, seed_base)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        track = self.tracks[i % len(self.tracks)]
        snippet = render_snippet(track, self.seed_base + i + self.epoch * self.length, self.config)
        audio = torch.from_numpy(snippet.mix.mean(axis=1))  # mono
        return audio, torch.from_numpy(class_vector(snippet.labels["classes"]))
