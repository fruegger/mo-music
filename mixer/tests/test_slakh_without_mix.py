import numpy as np
import soundfile as sf
import yaml
from mixer import iter_tracks


def make_track(root, with_mix):
    track = root / "train" / "Track00001"
    (track / "stems").mkdir(parents=True)
    rng = np.random.default_rng(0)
    stems = {}
    for sid, cls in (("S00", "Piano"), ("S01", "Bass")):
        sf.write(track / "stems" / f"{sid}.flac", rng.normal(0, 0.1, 44100).astype("float32"), 44100)
        stems[sid] = {"audio_rendered": True, "inst_class": cls, "is_drum": False, "midi_program_name": cls}
    (track / "metadata.yaml").write_text(yaml.safe_dump({"stems": stems}), encoding="utf-8")
    if with_mix:
        sf.write(track / "mix.flac", rng.normal(0, 0.1, 44100).astype("float32"), 44100)


def test_slakh_track_loads_without_mix_flac(tmp_path):
    make_track(tmp_path / "with", True)
    make_track(tmp_path / "without", False)
    (a,) = iter_tracks("slakh2100", tmp_path / "with", "train")
    (b,) = iter_tracks("slakh2100", tmp_path / "without", "train")
    assert (b.sample_rate, b.n_frames) == (a.sample_rate, a.n_frames) == (44100, 44100)
    assert [s.name for s in b.stems] == ["S00", "S01"] and b.stems[0].load(0, 1000).shape[0] == 1000


def test_second_listing_reads_the_index_not_the_files(tmp_path):
    make_track(tmp_path, True)
    first = list(iter_tracks("slakh2100", tmp_path, "train"))
    index = tmp_path / "train" / ".mixer-index.json"
    assert index.exists()
    (tmp_path / "train" / "Track00001" / "metadata.yaml").unlink()  # a rescan would fail now
    (b,) = iter_tracks("slakh2100", tmp_path, "train")
    assert (b.name, b.sample_rate, b.n_frames) == (first[0].name, 44100, 44100)
    assert [(s.name, s.label, s.source_class) for s in b.stems] == [(s.name, s.label, s.source_class) for s in first[0].stems]
    assert b.stems[0].load(0, 1000).shape[0] == 1000
