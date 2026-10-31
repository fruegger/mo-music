"""The shared instrument vocabulary.

Slakh2100's 13 `inst_class` values (spelled exactly as in its metadata.yaml,
including the odd "Strings (continued)" group) plus "Vocals". MUSDB18's stems
map onto it; its `other` stem is a mixture of many instruments, so it has no
class (`None`): it must never be read as a negative for any instrument.
"""

CLASSES = [
    "Bass",
    "Brass",
    "Chromatic Percussion",
    "Drums",
    "Guitar",
    "Organ",
    "Piano",
    "Pipe",
    "Reed",
    "Strings",
    "Strings (continued)",
    "Synth Lead",
    "Synth Pad",
    "Vocals",
]

SLAKH_CLASSES = {c: c for c in CLASSES if c != "Vocals"}

MUSDB_CLASSES = {"drums": "Drums", "bass": "Bass", "vocals": "Vocals", "other": None}

# Classes MUSDB18's labelled stems cover exactly. Everything else may be
# hiding in its `other` stem, so while `other` is audible the truth for those
# classes is unknown.
EXACT_IN_MUSDB = ["Bass", "Drums", "Vocals"]
