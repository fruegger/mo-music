"""Source-separation targets: the mixer's stems grouped into the sources a model is asked to produce.

Two source sets:

* ``musdb4``: drums, bass, vocals, other (MUSDB18's convention). Every stem has
  a place (an unknown mixture such as MUSDB18's own ``other`` goes to
  ``other``), so the sources always sum to the mix and every target is known.
* ``classes``: one source per instrument class (``mixer.CLASSES``). A stem with
  no class (MUSDB18 ``other``) belongs to no source, so while one is audible
  the classes it could contain have no known target: they are masked, as in the
  instrument-recognition loss.
"""

import numpy as np
from mixer import CLASSES
from mixer.vocab import EXACT_IN_MUSDB

MUSDB4 = ["drums", "bass", "vocals", "other"]
_MUSDB4_OF = {"Drums": "drums", "Bass": "bass", "Vocals": "vocals"}

SOURCE_SETS = {"musdb4": MUSDB4, "classes": list(CLASSES)}


def source_names(source_set: str) -> list[str]:
    return SOURCE_SETS[source_set]


def group_sources(stems: np.ndarray, labels: dict, source_set: str) -> tuple[np.ndarray, np.ndarray]:
    """`stems`: `(n_stems, frames, channels)` as heard in the mix (`Snippet.stems`),
    `labels`: the matching label file. Returns `(sources, mask)` with sources
    `(n_sources, frames, channels)` (silence where nothing contributes) and mask
    `(n_sources,)`, 1 where the target is known."""
    names = source_names(source_set)
    index = {name: i for i, name in enumerate(names)}
    sources = np.zeros((len(names), *stems.shape[1:]), dtype=np.float32)
    mask = np.ones(len(names), dtype=np.float32)

    for info, audio in zip(labels["stems"], stems):
        if source_set == "musdb4":
            target = _MUSDB4_OF.get(info["class"], "other")
        else:
            target = info["class"]  # None for an unknown mixture: no source
        if target is not None:
            sources[index[target]] += audio

    if source_set == "classes" and labels["ambiguous"]:
        mask[:] = 0.0
        for name in EXACT_IN_MUSDB:
            mask[index[name]] = 1.0
    return sources, mask
