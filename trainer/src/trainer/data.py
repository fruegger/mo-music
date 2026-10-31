"""Torch dataset over the mixer: snippets are rendered on the fly."""

import numpy as np
import torch
from mixer import CLASSES, MixConfig, iter_tracks, render_snippet
from mixer.vocab import EXACT_IN_MUSDB
from torch.utils.data import Dataset

from .sources import group_sources

CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}


def class_vector(classes) -> np.ndarray:
    v = np.zeros(len(CLASSES), dtype=np.float32)
    for c in classes:
        v[CLASS_INDEX[c]] = 1.0
    return v


def loss_mask(labels: dict) -> np.ndarray:
    """1 where the target is known. While an unlabelled mixture stem (MUSDB18
    `other`) is audible, any class it might contain is unknown, so only the
    classes MUSDB18 labels exactly keep their target."""
    mask = np.ones(len(CLASSES), dtype=np.float32)
    if labels["ambiguous"]:
        mask[:] = 0.0
        for c in EXACT_IN_MUSDB:
            mask[CLASS_INDEX[c]] = 1.0
    return mask


def as_stereo(audio: np.ndarray) -> np.ndarray:
    """`(..., frames, channels)` -> two channels: Slakh2100 is mono, MUSDB18 stereo."""
    return np.repeat(audio, 2, axis=-1) if audio.shape[-1] == 1 else audio


class SnippetDataset(Dataset):
    """`length` snippets drawn from the given tracks.

    Item `i` uses track `i % n_tracks` and seed `seed_base + i + epoch * length`
    (set `epoch` via `set_epoch` for training; leave at 0 for fixed validation
    sets). An item is a dict of tensors, by task:

    * `instruments`: `mono` `(T,)`, `instruments` `(n_classes,)`, `instruments_mask`
    * `separation`: `mix` `(2, T)`, `sources` `(n_sources, 2, T)`, `sources_mask`
      `(n_sources,)`; the sources of `source_set` (see `trainer.sources`)
    """

    def __init__(self, tracks, config: MixConfig, length: int, seed_base: int = 0,
                 tasks=("instruments",), source_set: str = "musdb4"):
        self.tracks = list(tracks)
        self.config = config
        self.length = length
        self.seed_base = seed_base
        self.tasks = tuple(tasks)
        self.source_set = source_set
        self.epoch = 0

    @classmethod
    def from_dataset(cls, dataset, split, config, length, seed_base=0, **kwargs):
        return cls(iter_tracks(dataset.name, dataset.path, split), config, length, seed_base, **kwargs)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> dict:
        track = self.tracks[i % len(self.tracks)]
        snippet = render_snippet(track, self.seed_base + i + self.epoch * self.length, self.config)
        labels = snippet.labels
        item = {}
        if "instruments" in self.tasks:
            item["mono"] = torch.from_numpy(snippet.mix.mean(axis=1))
            item["instruments"] = torch.from_numpy(class_vector(labels["classes"]))
            item["instruments_mask"] = torch.from_numpy(loss_mask(labels))
        if "separation" in self.tasks:
            sources, mask = group_sources(as_stereo(snippet.stems), labels, self.source_set)
            item["mix"] = torch.from_numpy(np.ascontiguousarray(as_stereo(snippet.mix).T))
            item["sources"] = torch.from_numpy(np.ascontiguousarray(sources.transpose(0, 2, 1)))
            item["sources_mask"] = torch.from_numpy(mask)
        return item
