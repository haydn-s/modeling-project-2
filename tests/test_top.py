import hashlib
import json
import os

import pandas as pd
import pytest

from trial_outcome.data import top
from trial_outcome.data.ctgov import write_jsonl
from trial_outcome.labeling import build_labels

HEADER = "nctid,status,why_stop,label,phase,diseases,icdcodes,drugs,smiless,criteria\n"


def row(
    nct_id: str,
    label: int,
    phase: str,
    status: str = "completed",
    why: str = "",
    icd: str = "[]",
    criteria: str = "Inclusion Criteria: adults",
) -> str:
    icd, criteria = (f'"{cell.replace('"', '""')}"' for cell in (icd, criteria))
    return f"{nct_id},{status},{why},{label},{phase},[],{icd},[],[],{criteria}\n"


# Multi-phase trials appear in each phase's files: NCT00000001 is phase 1/2 and
# NCT00000002 is phase 2/3. NCT00000003 is observational in the `study_records` fixture.
TOP_ROWS = {
    ("I", "train"): [row("NCT00000001", 1, "phase 1/phase 2")],
    ("I", "valid"): [],
    ("I", "test"): [row("NCT00000005", 0, "phase 1", "terminated", "lack of efficacy")],
    ("II", "train"): [
        row("NCT00000001", 1, "phase 1/phase 2"),
        row("NCT00000002", 0, "phase 2/phase 3"),
    ],
    ("II", "valid"): [],
    ("II", "test"): [row("NCT00000006", 1, "phase 2")],
    ("III", "train"): [row("NCT00000002", 0, "phase 2/phase 3")],
    ("III", "valid"): [],
    ("III", "test"): [row("NCT00000003", 1, "phase 3")],
}


def write_top(directory, rows=TOP_ROWS) -> dict[str, str]:
    """Write fixture TOP files and return their checksums."""
    directory.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for (phase, split), lines in rows.items():
        content = (HEADER + "".join(lines)).encode()
        name = top.top_filename(phase, split)
        (directory / name).write_bytes(content)
        checksums[name] = hashlib.sha256(content).hexdigest()
    return checksums


@pytest.fixture
def top_dir(tmp_path, monkeypatch):
    directory = tmp_path / "top"
    monkeypatch.setattr(top, "TOP_SHA256", write_top(directory))
    return directory


def test_load_top_combines_phases_and_splits(top_dir):
    table = top.load_top(top_dir)
    assert len(table) == 7
    assert table.columns.tolist() == ["phase", *top.LABEL_COLUMNS]
    phase_ii = table[table["phase"].eq("II")].set_index("nct_id")
    assert phase_ii["split"].to_dict() == {
        "NCT00000001": "train",
        "NCT00000002": "train",
        "NCT00000006": "test",
    }
    test_row = table.set_index(["phase", "nct_id"]).loc[("I", "NCT00000005")]
    assert test_row["why_stop"] == "lack of efficacy"
    assert pd.isna(table.set_index(["phase", "nct_id"]).loc[("I", "NCT00000001"), "why_stop"])


def test_label_files_follow_the_labels_contract(top_dir, tmp_path):
    out = tmp_path / "labels"
    paths = top.write_label_files(top.load_top(top_dir), out)
    labels = build_labels(paths["II"])
    assert labels.columns.tolist() == ["nct_id", "label"]
    assert labels.set_index("nct_id")["label"].to_dict() == {
        "NCT00000001": 1,
        "NCT00000002": 0,
        "NCT00000006": 1,
    }
    assert (out / "nct_ids.txt").read_text().split() == [
        "NCT00000001",
        "NCT00000002",
        "NCT00000003",
        "NCT00000005",
        "NCT00000006",
    ]
    assert top.load_splits(paths["I"]).to_dict() == {"NCT00000001": "train", "NCT00000005": "test"}


def test_changed_file_is_rejected(top_dir):
    path = top_dir / "phase_III_train.csv"
    path.write_text(path.read_text() + row("NCT00000004", 1, "phase 3"))
    with pytest.raises(ValueError, match="does not match the pinned TOP commit"):
        top.load_top(top_dir)


def test_missing_file_names_the_labels_command(top_dir):
    (top_dir / "phase_II_valid.csv").unlink()
    with pytest.raises(FileNotFoundError, match=r"python -m trial_outcome\.data\.top labels`"):
        top.load_top(top_dir)


@pytest.mark.parametrize(
    "change,message",
    [
        (  # the phase 1/2 trial has a different label in its phase II file
            {("II", "train"): [row("NCT00000001", 0, "phase 1/phase 2")]},
            "different labels",
        ),
        (  # the phase 1/2 trial is in phase II's test split but phase I's train split
            {("II", "train"): [], ("II", "test"): [row("NCT00000001", 1, "phase 1/phase 2")]},
            "test for one phase",
        ),
        ({("III", "valid"): [row("NCT00000002", 0, "phase 2/phase 3")]}, "duplicate nct_id"),
        ({("III", "test"): [row("NCT00000009", 2, "phase 3")]}, "only 0 and 1"),
    ],
)
def test_inconsistent_top_files_fail(tmp_path, monkeypatch, change, message):
    directory = tmp_path / "top"
    monkeypatch.setattr(top, "TOP_SHA256", write_top(directory, {**TOP_ROWS, **change}))
    with pytest.raises(ValueError, match=message):
        top.load_top(directory)


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        pass


class FakeSession:
    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.requested: list[str] = []

    def get(self, url: str, timeout: float) -> FakeResponse:
        name = url.removeprefix(top.TOP_BASE_URL)
        self.requested.append(name)
        return FakeResponse(self.files[name])


def test_download_verifies_checksums_and_skips_valid_files(top_dir, tmp_path):
    files = {path.name: path.read_bytes() for path in top_dir.iterdir()}
    dest = tmp_path / "download"
    dest.mkdir()
    (dest / "phase_I_test.csv").write_bytes(files["phase_I_test.csv"])
    (dest / "phase_I_train.csv").write_text("stale copy")
    session = FakeSession(files)

    top.download_top(dest, session=session)

    assert "phase_I_test.csv" not in session.requested
    assert "phase_I_train.csv" in session.requested
    assert len(session.requested) == 8
    assert {path.name: path.read_bytes() for path in dest.iterdir()} == files


def test_download_rejects_unexpected_content(top_dir, tmp_path):
    files = {path.name: path.read_bytes() for path in top_dir.iterdir()}
    files["phase_II_test.csv"] = b"tampered"
    dest = tmp_path / "download"
    with pytest.raises(ValueError, match="phase_II_test.csv: downloaded checksum"):
        top.download_top(dest, session=FakeSession(files))
    assert not (dest / "phase_II_test.csv").exists()
    assert not [path for path in dest.iterdir() if path.name.startswith(".")]


@pytest.mark.parametrize(
    "content", ["nct_id,split\nNCT00000001,holdout\n", "nct_id,split\nNCT00000002,\n"]
)
def test_load_splits_validates_values(tmp_path, content):
    path = tmp_path / "labels.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match="split must be one of"):
        top.load_splits(path)


# ICD codes: TOP's cell is a list of list literals, one per disease name.
ICD_CELL = "[\"['C50.911', 'D63.1']\", \"['C50.912']\"]"


def test_load_top_features_parses_icd_codes_and_criteria(tmp_path, monkeypatch):
    directory = tmp_path / "top"
    rows = {
        **TOP_ROWS,
        ("I", "test"): [row("NCT00000005", 0, "phase 1", icd=ICD_CELL, criteria="  ")],
    }
    monkeypatch.setattr(top, "TOP_SHA256", write_top(directory, rows))

    features = top.load_top_features(directory)

    assert features.index.tolist() == [f"NCT0000000{i}" for i in (1, 2, 3, 5, 6)]
    assert features.dtypes.astype(str).to_dict() == top.TOP_FEATURE_DTYPES
    trial = features.loc["NCT00000005"]
    assert trial["icd_chapters"] == "blood_immune|neoplasms"
    assert trial["icd_categories"] == "C50|D63"
    assert pd.isna(trial["criteria"])  # blank criteria are missing
    other = features.loc["NCT00000001"]
    assert (other["icd_chapters"], other["icd_categories"]) == ("", "")
    assert other["criteria"] == "Inclusion Criteria: adults"


@pytest.mark.parametrize(
    "change,message",
    [
        (  # the phase 1/2 trial has different criteria in its phase II file
            {("II", "train"): [row("NCT00000001", 1, "phase 1/phase 2", criteria="children")]},
            "different ICD codes or criteria",
        ),
        ({("II", "test"): [row("NCT00000006", 1, "phase 2", icd="['C50")]}, "NCT00000006"),
        ({("II", "test"): [row("NCT00000006", 1, "phase 2", icd="[\"['X']\"]")]}, "ICD-10"),
    ],
)
def test_load_top_features_rejects_bad_files(tmp_path, monkeypatch, change, message):
    directory = tmp_path / "top"
    monkeypatch.setattr(top, "TOP_SHA256", write_top(directory, {**TOP_ROWS, **change}))
    with pytest.raises(ValueError, match=message):
        top.load_top_features(directory)


@pytest.mark.parametrize("with_top_features", [True, False])
def test_build_top_datasets_writes_one_dataset_per_phase(
    top_dir, tmp_path, study_records, with_top_features
):
    labels_dir = tmp_path / "labels"
    top.write_label_files(top.load_top(top_dir), labels_dir)
    top_features = top.write_top_features(top.load_top_features(top_dir), tmp_path / "top.parquet")
    raw = tmp_path / "raw.jsonl"
    write_jsonl(study_records, raw)
    root = tmp_path / "processed"

    manifests = top.build_top_datasets(
        raw, root, labels_dir, top_features if with_top_features else None
    )

    phase_ii = pd.read_parquet(root / "phase_II" / "dataset.parquet")
    assert phase_ii["label"].to_dict() == {"NCT00000001": 1, "NCT00000002": 0}
    assert not {"split", "status", "why_stop", "top_phase"} & set(phase_ii.columns)
    assert (set(top.TOP_FEATURE_DTYPES) <= set(phase_ii.columns)) == with_top_features
    assert ("extra_features" in manifests["II"]["inputs"]) == with_top_features
    phase_iii = pd.read_parquet(root / "phase_III" / "dataset.parquet")
    assert phase_iii["label"].to_dict() == {"NCT00000002": 0}
    unmatched = pd.read_parquet(root / "phase_III" / "unmatched_labels.parquet")
    assert unmatched["reason"].to_dict() == {"NCT00000003": "excluded_from_cohort"}
    assert manifests["I"]["counts"]["unmatched_labels"] == 1  # NCT00000005 is not in raw
    assert "TOP benchmark, phase II" in manifests["II"]["configuration"]["label_description"]
    with pytest.raises(FileExistsError, match="already exist"):
        top.build_top_datasets(raw, root, labels_dir)


def test_cli_writes_labels_without_downloading_valid_files(top_dir, tmp_path, capsys):
    out, features = tmp_path / "labels", tmp_path / "top_features.parquet"
    top.main(
        [
            "labels",
            "--top-dir",
            str(top_dir),
            "--out-dir",
            str(out),
            "--features-out",
            str(features),
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["top_features"] == str(features)
    assert pd.read_parquet(features).index.tolist() == top.load_top_features(top_dir).index.tolist()
    assert summary["unique_trials"] == 5
    assert summary["phase_II"]["train"] == {"trials": 2, "success_rate": 0.5}
    assert sorted(path.name for path in out.iterdir()) == [
        "nct_ids.txt",
        "phase_I.csv",
        "phase_II.csv",
        "phase_III.csv",
    ]


def test_rewriting_unchanged_labels_and_features_keeps_their_timestamps(top_dir, tmp_path):
    out, features = tmp_path / "labels", tmp_path / "top_features.parquet"
    top.write_label_files(top.load_top(top_dir), out)
    top.write_top_features(top.load_top_features(top_dir), features)
    paths = [*out.iterdir(), features]
    for path in paths:
        os.utime(path, ns=(0, 0))

    top.write_label_files(top.load_top(top_dir), out)
    top.write_top_features(top.load_top_features(top_dir), features)

    assert [path.stat().st_mtime_ns for path in paths] == [0] * len(paths)
