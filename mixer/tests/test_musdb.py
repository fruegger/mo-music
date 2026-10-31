import json
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")

from mixer.mix import MixConfig, render_snippet  # noqa: E402
from mixer.sources import MUSDB_STEMS, SAMPLE_RATE, decode_musdb, iter_tracks, prepare_musdb  # noqa: E402

REAL = Path(r"C:/workarea/MUSDB18")
NOTE_TIMES = {"drums": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "bass": [1.5, 3.5, 5.5, 7.5], "other": [2.5, 6.5], "vocals": [4.5]}


def stem_audio(times, seconds=10):
    n = SAMPLE_RATE * seconds
    out = np.zeros(n, dtype=np.float32)
    t = np.arange(n) / SAMPLE_RATE
    for start in times:
        i = int(start * SAMPLE_RATE)
        out[i:] += (0.4 * np.exp(-8 * t[: n - i]) * np.sin(2 * np.pi * 200 * t[: n - i])).astype(np.float32)
    return np.stack([out, out])  # (channels, samples)


def write_stem_mp4(path: Path):
    """A MUSDB18-shaped file: mixture, drums, bass, other, vocals as five AAC streams."""
    streams = {"mixture": sum(stem_audio(t) for t in NOTE_TIMES.values())}
    streams.update({name: stem_audio(NOTE_TIMES[name]) for name in MUSDB_STEMS})
    with av.open(str(path), "w") as container:
        outputs = {}
        for name in streams:
            stream = container.add_stream("aac", rate=SAMPLE_RATE)
            stream.layout = "stereo"
            outputs[name] = stream
        for name, audio in streams.items():
            stream = outputs[name]
            for lo in range(0, audio.shape[1], 1024):
                chunk = np.ascontiguousarray(audio[:, lo : lo + 1024])
                frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="stereo")
                frame.sample_rate = SAMPLE_RATE
                frame.pts = lo
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)


@pytest.fixture(scope="module")
def musdb_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("musdb")
    (root / "train").mkdir()
    write_stem_mp4(root / "train" / "Fake Artist - Fake Song.stem.mp4")
    return root


def test_decode_reads_the_four_stems(musdb_root):
    stems, rate = decode_musdb(musdb_root / "train" / "Fake Artist - Fake Song.stem.mp4")
    assert rate == SAMPLE_RATE and set(stems) == set(MUSDB_STEMS)
    assert all(a.ndim == 2 and a.shape[1] == 2 and abs(len(a) - 10 * SAMPLE_RATE) < 4096 for a in stems.values())
    assert np.abs(stems["drums"]).max() > 0.1 and np.abs(stems["vocals"]).max() > 0.1


def test_prepare_writes_flac_stems_and_estimated_onsets(musdb_root, tmp_path):
    assert prepare_musdb(musdb_root, tmp_path, splits=("train",)) == 1
    track_dir = tmp_path / "train" / "Fake Artist - Fake Song"
    assert {p.name for p in track_dir.iterdir()} == {"drums.flac", "bass.flac", "other.flac", "vocals.flac", "onsets.json"}
    meta = json.loads((track_dir / "onsets.json").read_text())
    assert meta["sample_rate"] == SAMPLE_RATE and set(meta["onsets"]) == set(MUSDB_STEMS)
    drums = np.array(meta["onsets"]["drums"])
    expected = np.array(NOTE_TIMES["drums"])
    assert len(drums) == len(expected)
    # AAC adds ~25 ms of encoder delay to every stream; the estimate must be within that and the detector's own error.
    assert np.abs(np.sort(drums) - expected).max() < 0.07


def test_prepared_and_raw_layouts_agree(musdb_root, tmp_path):
    prepare_musdb(musdb_root, tmp_path, splits=("train",))
    raw = next(iter_tracks("musdb18", musdb_root, "train"))
    prepared = next(iter_tracks("musdb18", tmp_path, "train"))
    assert raw.name == prepared.name and raw.sample_rate == prepared.sample_rate
    assert [s.name for s in raw.stems] == [s.name for s in prepared.stems] == MUSDB_STEMS
    for a, b in zip(raw.stems, prepared.stems):
        assert np.allclose(a.onsets(), b.onsets(), atol=0.001)


def test_snippets_carry_estimated_onsets_and_the_provenance(musdb_root, tmp_path):
    prepare_musdb(musdb_root, tmp_path, splits=("train",))
    track = next(iter_tracks("musdb18", tmp_path, "train"))
    config = MixConfig(seconds=6.0, p_stem_dropped=0.0, effects=False, gain_db_range=(0.0, 0.0))
    labels = render_snippet(track, 1, config).labels
    assert labels["times"]["onsets_estimated"] is True and labels["times"]["onsets_complete"] is True
    assert labels["ambiguous"] is True  # `other` is audible
    sources = {s["stem"]: s["onset_source"] for s in labels["stems"]}
    assert set(sources.values()) == {"estimated"}
    assert "other" not in str(labels["times"]["onsets_by_class"])  # no class for the unknown mixture
    drums = next(s for s in labels["stems"] if s["stem"] == "drums")["onsets"]
    assert len(drums) >= 4 and all(0.0 <= o < 6.0 for o in drums)


@pytest.mark.skipif(not (REAL / "train").exists(), reason="MUSDB18 not available")
def test_real_track_round_trip(tmp_path):
    assert prepare_musdb(REAL, tmp_path, splits=("train",), limit=1) == 1
    track = next(iter_tracks("musdb18", tmp_path, "train"))
    labels = render_snippet(track, 0, MixConfig(effects=False)).labels
    assert labels["times"]["onsets_estimated"] and len(labels["times"]["onsets"]) > 5
