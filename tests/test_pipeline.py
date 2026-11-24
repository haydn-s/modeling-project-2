import hashlib
import json

import pandas as pd
import pytest

from trial_outcome.data.ctgov import write_jsonl
from trial_outcome.data.pipeline import build_dataset, main


@pytest.fixture
def raw_path(tmp_path, study_records):
    path = tmp_path / "raw.jsonl"
    write_jsonl(study_records, path)
    return path


def test_unlabeled_build_filters_cohort_and_tracks_provenance(tmp_path, raw_path):
    output = tmp_path / "processed"
    raw_before = raw_path.read_bytes()
    manifest = build_dataset(raw_path, output)
    assert manifest["counts"] == {
        "raw_studies": 3,
        "included_studies": 2,
        "excluded_studies": 1,
    }
    assert manifest == json.loads((output / "manifest.json").read_text())
    assert manifest["inputs"]["raw"]["sha256"] == hashlib.sha256(raw_before).hexdigest()
    assert raw_path.read_bytes() == raw_before
    assert not (output / "dataset.parquet").exists()
    features = pd.read_parquet(output / "features.parquet")
    assert features.index.tolist() == ["NCT00000001", "NCT00000002"]
    assert "label" not in features
    audit = pd.read_parquet(output / "cohort_audit.parquet")
    assert not audit.loc["NCT00000003", "included"]
    assert audit.loc["NCT00000003", "exclusion_reason"] == "study_type_mismatch_or_missing"
    assert audit.loc["NCT00000001", "overall_status"] == "COMPLETED"
    for name, info in manifest["outputs"].items():
        assert info["sha256"] == hashlib.sha256((output / name).read_bytes()).hexdigest()
        assert info["rows"] == len(pd.read_parquet(output / name))


def test_labels_join_by_id_with_unmatched_and_unlabeled_counts(tmp_path, raw_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("nct_id,label\nNCT00000004,1\nNCT00000003,0\nNCT00000002,1\n")
    output = tmp_path / "labeled"
    manifest = build_dataset(raw_path, output, labels_path=labels, label_description="Test labels")
    dataset = pd.read_parquet(output / "dataset.parquet")
    assert dataset.index.tolist() == ["NCT00000002"]
    # A terminated study can have label 1: labels are supplied, not inferred from status.
    assert dataset["label"].tolist() == [1]
    assert manifest["counts"]["unlabeled_studies"] == 1
    assert manifest["counts"]["unmatched_labels"] == 2
    assert manifest["counts"]["success_labels"] == 1
    unmatched = pd.read_parquet(output / "unmatched_labels.parquet")
    assert unmatched["reason"].to_dict() == {
        "NCT00000003": "excluded_from_cohort",
        "NCT00000004": "not_in_raw",
    }


def test_rebuild_is_deterministic_and_never_overwrites(tmp_path, raw_path):
    output = tmp_path / "first"
    first = build_dataset(raw_path, output, study_type="ANY")
    second = build_dataset(raw_path, tmp_path / "second", study_type="ANY")
    assert first["outputs"] == second["outputs"]
    with pytest.raises(FileExistsError):
        build_dataset(raw_path, output)
    assert json.loads((output / "manifest.json").read_text()) == first


def test_failed_write_does_not_publish_partial_output(tmp_path, raw_path, monkeypatch):
    original = pd.DataFrame.to_parquet
    calls = 0

    def fail_second_write(frame, path, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        return original(frame, path, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_second_write)
    with pytest.raises(OSError, match="disk full"):
        build_dataset(raw_path, tmp_path / "processed")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["raw.jsonl"]


@pytest.mark.parametrize(
    "body,message",
    [
        ("", "no studies"),
        ("{}\n", "missing nct_id"),
        ("not json\n", "invalid JSON"),
        (
            '{"protocolSection":{"identificationModule":{"nctId":"NCT00000001"}}}\n',
            "no studies remain",
        ),
    ],
)
def test_bad_inputs_publish_nothing(tmp_path, body, message):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(body)
    with pytest.raises(ValueError, match=message):
        build_dataset(raw, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_nonoverlapping_labels_fail(tmp_path, raw_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("nct_id,label\nNCT99999999,0\n")
    with pytest.raises(ValueError, match="no labels match"):
        build_dataset(raw_path, tmp_path / "out", labels_path=labels, label_description="Test")
    assert not (tmp_path / "out").exists()


def test_label_source_description_is_required(tmp_path, raw_path):
    with pytest.raises(ValueError, match="label_description is required"):
        build_dataset(raw_path, tmp_path / "out", labels_path=tmp_path / "labels.csv")


def test_cli_builds_and_reports_errors(tmp_path, raw_path, capsys):
    args = ["--raw", str(raw_path), "--output-dir", str(tmp_path / "out")]
    main(args)
    assert json.loads(capsys.readouterr().out)["included_studies"] == 2
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert "already exists" in capsys.readouterr().err
