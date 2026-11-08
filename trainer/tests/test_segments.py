"""Hand labels from the listening pages: loaders, and exact targets for an unlabelled MUSDB18 stem over a labelled stretch."""

import pytest
import torch
from helpers import synthetic_track
from mixer import MixConfig, render_snippet

from trainer.data import SnippetDataset
from trainer.segments import exact_source, load_solo_windows, load_stem_labels, parse_interruptions
from trainer.train import SeparationTask

STEMS = "split,song,stem,instrument,verdict,keys_clean_pct,tagger_top_level,bar_threshold,interruptions\n"
SEGS = "split,song,stem,instrument,start_s,end_s,verdict\n"


def test_interruptions_are_parsed():
    assert parse_interruptions('3.0-5.5:guitar;10.0-12.0:?') == [(3.0, 5.5), (10.0, 12.0)] and parse_interruptions("") == []


def test_stem_labels_keep_all_and_mostly_only(tmp_path):
    f = tmp_path / "stems.csv"
    f.write_text(STEMS + 'train,A,other,piano,all,90,0.7,0.3,"1.0-2.0:guitar"\ntrain,B,other,piano,mostly,90,0.7,0.3,"4.0-6.0:strings"\n'
                 'train,C,other,piano,some,50,0.5,0.2,""\ntrain,D,other,organ,no,0,0,0,""\n', encoding="utf-8")
    labels = load_stem_labels(f, "musdb6")
    assert set(labels) == {"A", "B"} and labels["A"] == (4, []) and labels["B"] == (4, [(4.0, 6.0)])  # keys is source 4; `all` ignores the gaps


def test_solo_windows_keep_solo_only(tmp_path):
    f = tmp_path / "segs.csv"
    f.write_text(SEGS + "train,A,other,piano,10.0,15.0,solo\ntrain,A,other,piano,20.0,25.0,mixed\ntrain,B,other,organ,3.0,8.0,solo\n", encoding="utf-8")
    assert load_solo_windows(f, "musdb6") == {"A": [(10.0, 15.0, 4)], "B": [(3.0, 8.0, 4)]}


def test_exact_source_rules():
    stem = (4, [(4.0, 6.0)])
    assert exact_source((0.0, 3.0), stem, None) == 4 and exact_source((5.0, 8.0), stem, None) is None  # touches an interruption
    assert exact_source((7.0, 9.0), None, [(6.0, 10.0, 3)]) == 3 and exact_source((5.0, 9.0), None, [(6.0, 10.0, 3)]) is None
    assert exact_source((5.0, 9.0), stem, [(5.0, 10.0, 3)]) == 3  # the solo window rescues a span the stem judgement rejects


def _item(ds_kwargs, name="Song A", seconds=1.0, seed_index=0):
    config = MixConfig(seconds=seconds, p_stem_dropped=0.0, effects=False)
    track = synthetic_track(name=name, with_unknown=True)
    track.dataset = "musdb18"
    return SnippetDataset([track], config, 8, tasks=("separation",), source_set="musdb6", **ds_kwargs)[seed_index]


def test_a_whole_stem_judgement_makes_the_targets_exact():
    plain = _item({})
    assert plain["group_mask"].tolist() == [0, 0, 0, 1, 1, 1]
    keys = _item({"exact_stems": {"Song A": (4, [])}})
    assert keys["sources_mask"].tolist() == [1.0] * 6 and keys["group_mask"].tolist() == [0.0] * 6
    assert torch.allclose(keys["sources"][4], plain["sources"][4] + plain["sources"][5]) and torch.allclose(keys["mix"], plain["mix"])
    assert _item({"exact_stems": {"Song A": (4, [])}}, name="Song B")["group_mask"].tolist() == [0, 0, 0, 1, 1, 1]  # not judged


def test_an_interruption_under_the_snippet_keeps_it_ambiguous():
    whole = (0.0, 100.0)
    assert _item({"exact_stems": {"Song A": (4, [whole])}})["group_mask"].tolist() == [0, 0, 0, 1, 1, 1]


def test_snippets_are_placed_inside_a_solo_window_and_get_exact_targets():
    window = (6.0, 9.0)
    kwargs = {"solo_windows": {"Song A": [(*window, 4)]}, "p_solo": 1.0}
    for i in range(6):
        item = _item(kwargs, seed_index=i)
        assert item["sources_mask"].tolist() == [1.0] * 6  # every snippet fell inside the window
    none = _item({"solo_windows": {"Song A": [(*window, 4)]}, "p_solo": 0.0}, seed_index=0)
    assert none["group_mask"].tolist() == [0, 0, 0, 1, 1, 1] or none["sources_mask"].min() == 1.0  # by chance only


def test_start_window_places_the_snippet_and_changes_nothing_else():
    track = synthetic_track(name="W", with_unknown=True)
    config = MixConfig(seconds=1.0, p_stem_dropped=0.0, effects=False)
    for seed in range(5):
        snip = render_snippet(track, seed, config, start_window=(6.0, 9.0))
        s = snip.labels["start_seconds"]
        assert 6.0 - 1e-3 <= s and s + snip.labels["source_seconds"] <= 9.0 + 1e-3
    short = render_snippet(track, 0, config, start_window=(6.0, 6.5))
    assert abs(short.labels["start_seconds"] - 6.0) < 1e-3  # a window shorter than the snippet: starts at its beginning
    plain = render_snippet(track, 0, config)
    assert plain.labels["seed"] == 0 and render_snippet(track, 0, config, start_window=None).labels["start_frame"] == plain.labels["start_frame"]


def test_the_separation_task_reads_the_label_files(tmp_path):
    stems, segs = tmp_path / "stems.csv", tmp_path / "segs.csv"
    stems.write_text(STEMS + 'train,A,other,piano,all,90,0.7,0.3,""\n', encoding="utf-8")
    segs.write_text(SEGS + "train,A,other,piano,10.0,15.0,solo\n", encoding="utf-8")
    cfg = {"separation": {"source_set": "musdb6", "stem_labels": str(stems), "solo_windows": str(segs), "p_solo": 0.4}}
    kw = SeparationTask(cfg).dataset_kwargs
    assert kw["exact_stems"] == {"A": (4, [])} and kw["solo_windows"] == {"A": [(10.0, 15.0, 4)]} and kw["p_solo"] == 0.4
    assert set(SeparationTask({"separation": {"source_set": "musdb6"}}).dataset_kwargs) == {"source_set"}
