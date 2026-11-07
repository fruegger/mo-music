import pytest
import torch
from helpers import synthetic_track
from mixer import MixConfig

from trainer.data import SnippetDataset
from trainer.tags import ABSENT, MAYBE, PRESENT, UNKNOWN, load_other_tags
from trainer.train import PresenceTask

SAMPLES = 20000
CFG = {"presence": {"source_set": "musdb6", "n_mels": 16, "present_rel_db": -40.0, "absent_rel_db": -60.0, "tag_weight": 0.5}}
NO_TAGS = [UNKNOWN] * 6


def amp(db):
    return 10 ** (db / 20)


def item(group_db=None, tags=None, individual_db=(-10.0, -10.0, -10.0, -120.0, -120.0, -120.0)):
    """A batch of one item with a mix at 0 dB. With `group_db` the item only knows the SUM of guitar, keys and other
    (carried in `other`, as the dataset does), at that level re the mix; otherwise all six levels are known."""
    levels = list(individual_db)
    if group_db is not None:
        levels[3:] = [-120.0, -120.0, group_db]
    sources = torch.stack([torch.full((2, SAMPLES), amp(db)) for db in levels]).unsqueeze(0)
    group_mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0] if group_db is not None else [0.0] * 6])
    batch = {"sources": sources, "mix": torch.full((1, 2, SAMPLES), 1.0), "group_mask": group_mask,
             "group_target": sources[:, 3:].sum(dim=1)}
    if tags is not None:
        batch["source_tags"] = torch.tensor([tags])
    return batch


def test_an_untagged_instrument_in_an_audible_sum_is_absent_a_tagged_one_present_and_an_unsure_one_left_out():
    labels, weights = PresenceTask(CFG).targets(item(-3.0, tags=[-1, -1, -1, PRESENT, ABSENT, MAYBE]))
    assert labels[0, 3:].tolist() == [1.0, 0.0, 0.0] and weights[0, 3:].tolist() == [0.5, 1.0, 0.0]
    assert weights[0, :3].tolist() == [1.0, 1.0, 1.0]  # drums, bass and vocals are known as usual


def test_an_inaudible_sum_means_all_three_are_absent_even_with_tags():
    labels, weights = PresenceTask(CFG).targets(item(-90.0, tags=[-1, -1, -1, PRESENT, PRESENT, PRESENT]))
    assert labels[0, 3:].tolist() == [0.0] * 3 and weights[0, 3:].tolist() == [1.0] * 3


def test_a_sum_in_the_ignored_zone_leaves_the_three_out():
    task = PresenceTask(CFG)
    labels, weights = task.targets(item(-50.0, tags=[-1, -1, -1, PRESENT, ABSENT, ABSENT]))
    assert weights[0, 3:].tolist() == [0.0] * 3
    assert task.group_weights(item(-50.0, tags=NO_TAGS)).tolist() == [0.0]


def test_without_tags_the_three_are_left_out_and_the_group_term_applies():
    task = PresenceTask(CFG)
    for batch in (item(-3.0), item(-3.0, tags=NO_TAGS)):
        assert task.targets(batch)[1][0, 3:].tolist() == [0.0] * 3
        assert task.group_weights(batch).tolist() == [1.0]
    assert task.group_weights(item(-3.0, tags=[-1, -1, -1, PRESENT, ABSENT, ABSENT])).tolist() == [0.0]  # the tags say it already
    assert task.group_weights(item(-90.0)).tolist() == [0.0]  # inaudible: the individual labels already say it


def test_an_item_with_all_six_known_is_unchanged_by_the_group_logic():
    task = PresenceTask(CFG)
    known = item(None, individual_db=(-10.0, -10.0, -10.0, -10.0, -120.0, -45.0))
    labels, weights = task.targets(known)
    assert labels[0].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0] and weights[0].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    assert task.group_weights(known).tolist() == [0.0]


def test_the_group_is_scored_as_one_source():
    task = PresenceTask(CFG)
    assert [t.tolist() for t in task.group_labels(item(-3.0))] == [[1.0], [1.0]]  # audible sum: present
    assert [t.tolist() for t in task.group_labels(item(-90.0))] == [[0.0], [1.0]]  # inaudible: absent
    assert task.group_labels(item(-50.0))[1].tolist() == [0.0]  # ignored zone
    known = item(None, individual_db=(-10.0, -10.0, -10.0, -120.0, -10.0, -120.0))
    assert [t.tolist() for t in task.group_labels(known)] == [[1.0], [1.0]]  # keys present


def test_the_group_term_trains_the_noisy_or_and_tags_replace_it():
    task = PresenceTask(CFG)
    torch.manual_seed(0)
    model = task.build()
    untagged, tagged = item(-3.0, tags=NO_TAGS), item(-3.0, tags=[-1, -1, -1, PRESENT, ABSENT, ABSENT])
    loss = task.loss(model, untagged, "cpu")
    loss.backward()
    assert loss > 0 and all(p.grad is not None for p in model.parameters())
    # the same item without the group term (its weight forced to zero) has a different loss
    plain = task.loss(model, {k: v for k, v in untagged.items() if k != "group_mask"}, "cpu")
    assert not torch.isclose(loss, plain)
    assert torch.isfinite(task.loss(model, tagged, "cpu"))


def test_evaluation_reports_the_group_next_to_the_sources():
    task = PresenceTask(CFG)
    model = task.build().eval()
    scores = task.evaluate(model, [item(-3.0), item(-90.0)], "cpu")
    assert set(scores["group"]) == {"precision", "recall", "f1"} and "guitar" in scores["f1"]
    assert "group" not in PresenceTask({"presence": {"source_set": "musdb4", "n_mels": 16}}).evaluate(
        PresenceTask({"presence": {"source_set": "musdb4", "n_mels": 16}}).build().eval(),
        [{"sources": torch.rand(2, 4, 2, SAMPLES), "mix": torch.rand(2, 2, SAMPLES)}], "cpu")


def write_csv(path, rows):
    path.write_text("split,song,instruments,dominant,unsure,notes\n" + "".join(f"{r}\n" for r in rows), encoding="utf-8-sig")
    return path


def test_tags_map_to_the_sources_and_an_empty_song_is_left_out(tmp_path):
    csv = write_csv(tmp_path / "tags.csv", [
        "train,Guitar And Keys,guitar+synth,,,", "train,Band,guitar+Piano+trumpet,,sax,", "train,Maybe,guitar,,organ+flute,",
        "train,Nothing,,,,silence", "train,Vocals Only,vocals,,,"])
    tags = load_other_tags(csv)
    assert tags["Guitar And Keys"] == [-1, -1, -1, PRESENT, PRESENT, ABSENT]
    assert tags["Band"] == [-1, -1, -1, PRESENT, PRESENT, PRESENT]  # trumpet, sax -> other
    assert tags["Maybe"] == [-1, -1, -1, PRESENT, MAYBE, MAYBE]
    assert tags["Vocals Only"] == [-1, -1, -1, ABSENT, ABSENT, ABSENT]  # vocals are not an `other` instrument
    assert "Nothing" not in tags


def test_a_tag_word_with_no_source_is_an_error_not_an_absent_instrument(tmp_path):
    with pytest.raises(ValueError, match="uitar"):
        load_other_tags(write_csv(tmp_path / "tags.csv", ["train,Typo,uitar+synth,,,"]))


def test_the_real_tag_file_loads():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "labels" / "musdb18-other-tags.csv"
    tags = load_other_tags(path)
    assert len(tags) >= 140 and all(len(v) == 6 for v in tags.values())


def test_items_carry_the_tags_of_their_song_only_for_musdb18_songs_and_only_when_asked():
    config = MixConfig(seconds=1.0, p_stem_dropped=0.0, effects=False)
    tags = {"Song A": [-1, -1, -1, PRESENT, ABSENT, ABSENT]}

    def render(name, dataset="musdb18", **kwargs):
        track = synthetic_track(name=name, with_unknown=True)
        track.dataset = dataset
        return SnippetDataset([track], config, 1, tasks=("separation",), source_set="musdb6", **kwargs)[0]

    assert render("Song A", source_tags=tags)["source_tags"].tolist() == tags["Song A"]
    assert render("Song B", source_tags=tags)["source_tags"].tolist() == [-1.0] * 6
    assert render("Song A", dataset="studio", source_tags=tags)["source_tags"].tolist() == [-1.0] * 6
    assert "source_tags" not in render("Song A")


def test_a_recogniser_run_loads_without_its_tag_file():
    task = PresenceTask({"presence": {**CFG["presence"], "tags": "does/not/exist.csv"}})
    assert task.build().pool == "mean" and task.names[3] == "guitar"  # inference never reads the tags
    with pytest.raises(FileNotFoundError):
        task.dataset_kwargs


def test_a_song_with_one_source_only_is_promoted_to_exact_targets():
    from trainer.tags import single_source

    assert single_source([-1, -1, -1, PRESENT, ABSENT, ABSENT]) == 3
    assert single_source([-1, -1, -1, ABSENT, PRESENT, ABSENT]) == 4
    assert single_source([-1, -1, -1, PRESENT, PRESENT, ABSENT]) is None  # two sources
    assert single_source([-1, -1, -1, PRESENT, MAYBE, ABSENT]) is None  # a maybe: we do not know
    assert single_source([-1, -1, -1, ABSENT, ABSENT, ABSENT]) is None
    assert single_source([-1] * 6) is None  # no tags


def test_promotion_moves_the_unlabelled_stem_to_its_source_and_makes_every_target_known():
    config = MixConfig(seconds=1.0, p_stem_dropped=0.0, effects=False)

    def render(state, promote=True, name="Song A"):
        track = synthetic_track(name=name, with_unknown=True)
        track.dataset = "musdb18"
        ds = SnippetDataset([track], config, 1, tasks=("separation",), source_set="musdb6",
                            source_tags={"Song A": state}, promote_single=promote)
        return ds[0]

    plain = render([-1, -1, -1, ABSENT, PRESENT, ABSENT], promote=False)
    keys = render([-1, -1, -1, ABSENT, PRESENT, ABSENT])
    assert plain["group_mask"].tolist() == [0, 0, 0, 1, 1, 1]
    assert keys["sources_mask"].tolist() == [1.0] * 6 and keys["group_mask"].tolist() == [0.0] * 6
    assert torch.allclose(keys["sources"][4], plain["sources"][4] + plain["sources"][5]) and keys["sources"][5].abs().max() == 0
    assert torch.allclose(keys["mix"], plain["mix"])  # only the labelling changes, not the audio
    other = render([-1, -1, -1, ABSENT, ABSENT, PRESENT])  # an other-only stem stays where it is
    assert other["sources_mask"].tolist() == [1.0] * 6 and torch.allclose(other["sources"][5], plain["sources"][5])
    mixed = render([-1, -1, -1, PRESENT, PRESENT, ABSENT])
    assert mixed["group_mask"].tolist() == [0, 0, 0, 1, 1, 1]  # two sources: still only the sum is known
    assert render([-1, -1, -1, ABSENT, PRESENT, ABSENT], name="Song B")["group_mask"].tolist() == [0, 0, 0, 1, 1, 1]  # not tagged


def test_a_promoted_item_gets_exact_individual_labels_from_the_levels():
    task = PresenceTask({"presence": {**CFG["presence"], "promote_single": True}})
    known = item(None, individual_db=(-10.0, -10.0, -10.0, -120.0, -10.0, -120.0), tags=[-1, -1, -1, ABSENT, PRESENT, ABSENT])
    labels, weights = task.targets(known)
    assert labels[0, 3:].tolist() == [0.0, 1.0, 0.0] and weights[0, 3:].tolist() == [1.0, 1.0, 1.0]  # no tag_weight discount
    assert task.group_weights(known).tolist() == [0.0]


def test_promotion_is_off_by_default_and_the_flag_reaches_the_dataset(tmp_path):
    csv = write_csv(tmp_path / "tags.csv", ["train,Song A,guitar,,,"])
    on = PresenceTask({"presence": {**CFG["presence"], "tags": str(csv), "promote_single": True}}).dataset_kwargs
    off = PresenceTask({"presence": {**CFG["presence"], "tags": str(csv)}}).dataset_kwargs
    assert on["promote_single"] is True and off["promote_single"] is False and on["source_tags"]["Song A"][3] == PRESENT


def test_the_separation_task_passes_the_tags_and_promotion_to_the_dataset(tmp_path):
    from trainer.train import SeparationTask
    csv = tmp_path / "t.csv"
    csv.write_text("split,song,instruments,dominant,unsure,notes\ntrain,A,guitar,,,\n", encoding="utf-8")
    cfg = {"separation": {"source_set": "musdb6"}}
    assert set(SeparationTask(cfg).dataset_kwargs) == {"source_set"}  # no tags: as before
    on = SeparationTask({"separation": {**cfg["separation"], "tags": str(csv), "promote_single": True}}).dataset_kwargs
    assert on["promote_single"] is True and on["source_tags"]["A"][3] == 1.0
    off = SeparationTask({"separation": {**cfg["separation"], "tags": str(csv)}}).dataset_kwargs
    assert off["promote_single"] is False
