"""Torch dataset over the mixer: snippets are rendered on the fly."""

import numpy as np
import torch
from mixer import CLASSES, MixConfig, iter_tracks, render_snippet
from mixer.vocab import EXACT_IN_MUSDB
from torch.utils.data import Dataset

from .sources import group_sources, source_names

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
    sets). The epoch lives in shared memory, so `set_epoch` reaches persistent
    DataLoader workers too.

    Tracks come either as `tracks` (any iterable; not picklable, so only for
    `num_workers=0`) or as `source=(dataset name, path, split)`, which each
    worker process turns into tracks itself on first use: this is what makes the
    dataset picklable for the spawn start method (Windows). An item is a dict of
    tensors, by task:

    * `instruments`: `mono` `(T,)`, `instruments` `(n_classes,)`, `instruments_mask`
    * `separation`: `mix` `(2, T)`, `sources` `(n_sources, 2, T)`, `sources_mask`
      `(n_sources,)`; the sources of `source_set` (see `trainer.sources`)
    """

    def __init__(self, tracks, config: MixConfig, length: int, seed_base: int = 0,
                 tasks=("instruments",), source_set: str = "musdb4", source=None):
        self._tracks = None if tracks is None else list(tracks)
        self.source = source
        self.config = config
        self.length = length
        self.seed_base = seed_base
        self.tasks = tuple(tasks)
        self.source_set = source_set
        self._epoch = torch.zeros(1, dtype=torch.long).share_memory_()

    @classmethod
    def from_dataset(cls, dataset, split, config, length, seed_base=0, **kwargs):
        """Lazy: the tracks are listed on first use, in whichever process that is."""
        return cls(None, config, length, seed_base, source=(dataset.name, str(dataset.path), split), **kwargs)

    @property
    def tracks(self) -> list:
        if self._tracks is None:
            name, path, split = self.source
            self._tracks = list(iter_tracks(name, path, split))
        return self._tracks

    def __getstate__(self):
        state = dict(self.__dict__)
        if self.source is not None:
            state["_tracks"] = None  # tracks hold closures; a worker lists its own
        return state

    @property
    def epoch(self) -> int:
        return int(self._epoch[0])

    def set_epoch(self, epoch: int) -> None:
        self._epoch[0] = epoch

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


class VocalMixDataset(Dataset):
    """Wraps a separation dataset that has no vocals (Slakh2100) and, with probability `p`,
    adds the vocals of a snippet from `vocals` (a MUSDB18 `SnippetDataset`) to the mix and to
    the `vocals` target. Both datasets must render the same snippet length at the same sample
    rate. Deterministic in (index, epoch); `set_epoch` reaches both wrapped datasets."""

    def __init__(self, base: Dataset, vocals: SnippetDataset, p: float = 0.5,
                 gain_db_range=(-6.0, 6.0), seed_base: int = 0, source_set: str = "musdb4", peak: float = 0.95):
        if len(vocals) < len(base):
            raise ValueError("the vocals dataset must be at least as long as the base dataset")
        self.base, self.vocals, self.p = base, vocals, p
        self.gain_db_range, self.seed_base, self.peak = tuple(gain_db_range), seed_base, peak
        self.vocal_index = source_names(source_set).index("vocals")

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)
        self.vocals.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        item = dict(self.base[i])
        rng = np.random.default_rng([self.seed_base, i, self.base.epoch])
        add, gain_db = rng.random() < self.p, rng.uniform(*self.gain_db_range)  # always two draws
        if not add:
            return item
        voice = self.vocals[i]["sources"][self.vocal_index] * 10.0 ** (gain_db / 20.0)
        mix, sources = item["mix"] + voice, item["sources"].clone()
        sources[self.vocal_index] += voice
        scale = min(1.0, self.peak / max(float(mix.abs().max()), 1e-9))
        item["mix"], item["sources"] = mix * scale, sources * scale
        item["sources_mask"] = item["sources_mask"].clone()
        item["sources_mask"][self.vocal_index] = 1.0
        return item
