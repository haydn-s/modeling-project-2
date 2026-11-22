"""TOP benchmark labels: download, verify, and convert them to the project's labels contract.

TOP (Trial Outcome Prediction) is the benchmark released with HINT (Fu et al., 2022,
*Patterns*). Its success/failure labels come from IQVIA. Each phase (I, II, III) has
train, valid, and test files, split by date: train and valid hold trials that completed
before 2014 (shuffled 90/10), and test holds trials that started in 2014 or later.
Trials registered as phase 1/2 or 2/3 appear in both phases' files.

    python -m trial_outcome.data.top labels
    python -m trial_outcome.data.ctgov --ids-file data/interim/top_labels/nct_ids.txt \\
        --out data/raw/ctgov_top.jsonl
    python -m trial_outcome.data.top datasets --raw data/raw/ctgov_top.jsonl \\
        --output-root data/processed/top

``labels`` writes one labels file per phase. Each keeps the TOP split and TOP's
``status``/``why_stop`` columns for analysis; the dataset build reads only ``nct_id`` and
``label``, so those columns never become features. TOP is licensed for non-commercial
academic use only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import closing, nullcontext
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
import requests

from trial_outcome.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from trial_outcome.data.ctgov import make_session
from trial_outcome.data.pipeline import build_dataset
from trial_outcome.data.validation import validate_nct_ids
from trial_outcome.labeling import validate_labels

TOP_REPO = "https://github.com/futianfan/clinical-trial-outcome-prediction"
TOP_COMMIT = "8dc0497f23fdb84e2905da7655924a91e6e79798"
TOP_BASE_URL = (
    "https://raw.githubusercontent.com/futianfan/clinical-trial-outcome-prediction/"
    f"{TOP_COMMIT}/data/"
)
PHASES = ("I", "II", "III")
SPLITS = ("train", "valid", "test")
# SHA-256 of each file at TOP_COMMIT; a mismatch means the data is not the pinned version.
TOP_SHA256 = {
    "phase_I_train.csv": "4db18b0f1469f215ed828c814acd6ed18ca3b7b49ae13483252553767f1a057a",
    "phase_I_valid.csv": "a170fca5f42781c98a3bf892bc81a46695c7c88810e4712555d6c68bd1765f44",
    "phase_I_test.csv": "85f614499bda8a8190152d99ff2799d06f8919cdd700572cd3f42bd86375d0f7",
    "phase_II_train.csv": "23e767816483b53ffad04c1e570d7ad4192e09a314ac0dd52375443c2afe2e39",
    "phase_II_valid.csv": "c5905bfc5b646e30321bd8e671d7bd946b8fb1dfe5c30a74bb29fa40e2a711e7",
    "phase_II_test.csv": "6e829c430ce1e397993dfc003f522f9e8089a6b13117800ff7943a252423ea91",
    "phase_III_train.csv": "b3e3c40a93d4c1ee3cd02a93d509f74dafafb5b219aac976ec62ac9c626df9ee",
    "phase_III_valid.csv": "8fc1d2a900e595685dc447dea0a2dd5bb9d500288b1c53166780da789f30cda2",
    "phase_III_test.csv": "537f0648c617bd4a25b05a98f4cd2ba77989a214655eb0be345a4dec5ee0db38",
}
LABEL_DESCRIPTION = (
    "TOP benchmark, phase {phase} (Fu et al. 2022, HINT; "
    f"{TOP_REPO} at commit {TOP_COMMIT[:7]}). "
    "label 1 = trial succeeded, 0 = trial failed, from IQVIA outcome labels as curated by TOP. "
    "TOP split by date: train/valid completed before 2014, test started in 2014 or later."
)
LABEL_COLUMNS = ["nct_id", "label", "split", "top_phase", "status", "why_stop"]

TOP_DIR = EXTERNAL_DATA_DIR / "top"
LABELS_DIR = INTERIM_DATA_DIR / "top_labels"


def top_filename(phase: str, split: str) -> str:
    """Return the TOP file name for one phase and split, e.g. ``phase_II_test.csv``."""
    return f"phase_{phase}_{split}.csv"


def labels_filename(phase: str) -> str:
    """Return the name of this project's labels file for one phase."""
    return f"phase_{phase}.csv"


def download_top(
    top_dir: str | Path = TOP_DIR,
    *,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> list[Path]:
    """Fetch the nine TOP files at the pinned commit and verify each checksum.

    Files already present with the expected checksum are not downloaded again. A file
    with an unexpected checksum is never written.
    """
    top_dir = Path(top_dir)
    top_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    context = nullcontext(session) if session is not None else closing(make_session())
    with context as client:
        for name, expected in TOP_SHA256.items():
            path = top_dir / name
            if not (path.exists() and _sha256(path) == expected):
                response = client.get(TOP_BASE_URL + name, timeout=timeout)
                response.raise_for_status()
                actual = hashlib.sha256(response.content).hexdigest()
                if actual != expected:
                    raise ValueError(f"{name}: downloaded checksum {actual} != pinned {expected}")
                _write_bytes_atomic(path, response.content)
            paths.append(path)
    return paths


def load_top(top_dir: str | Path = TOP_DIR) -> pd.DataFrame:
    """Combine the nine TOP files into one table with ``phase`` and ``split`` columns.

    Every file must match its pinned checksum. Within a phase, IDs must be unique and labels
    binary. A trial listed in two phases must have the same label in both, and may not be in
    the test split of one phase and the train/valid split of another, since that would leak
    test trials into training.
    """
    top_dir = Path(top_dir)
    frames = []
    for phase in PHASES:
        for split in SPLITS:
            name = top_filename(phase, split)
            path = top_dir / name
            if not path.exists():
                raise FileNotFoundError(
                    f"missing {path}; run `python -m trial_outcome.data.top labels`"
                )
            if _sha256(path) != TOP_SHA256[name]:
                raise ValueError(f"{path} does not match the pinned TOP commit; download it again")
            frame = pd.read_csv(
                path,
                usecols=["nctid", "label", "phase", "status", "why_stop"],
                dtype={"nctid": "string", "phase": "string", "status": "string"},
                keep_default_na=False,
                na_values={"label": [""]},
            )
            frame = frame.rename(columns={"nctid": "nct_id", "phase": "top_phase"})
            frame["why_stop"] = frame["why_stop"].astype("string").replace("", pd.NA)
            frames.append(frame.assign(phase=phase, split=split))
    top = pd.concat(frames, ignore_index=True)

    for phase, rows in top.groupby("phase"):
        try:
            validate_labels(rows)
        except ValueError as error:
            raise ValueError(f"TOP phase {phase}: {error}") from error
    top["label"] = top["label"].astype("int64")

    per_trial = top.groupby("nct_id").agg(
        labels=("label", "nunique"),
        in_test=("split", lambda s: s.eq("test").any()),
        in_learning=("split", lambda s: s.ne("test").any()),
    )
    conflicting = per_trial.index[per_trial["labels"] > 1].tolist()
    if conflicting:
        raise ValueError(f"trials have different labels in different phases: {conflicting[:5]}")
    leaking = per_trial.index[per_trial["in_test"] & per_trial["in_learning"]].tolist()
    if leaking:
        raise ValueError(
            f"trials are in test for one phase and train/valid for another: {leaking[:5]}"
        )
    return top[["phase", *LABEL_COLUMNS]]


def write_label_files(top: pd.DataFrame, out_dir: str | Path = LABELS_DIR) -> dict[str, Path]:
    """Write one labels file per phase plus ``nct_ids.txt`` listing every TOP trial once."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for phase in PHASES:
        rows = top.loc[top["phase"].eq(phase), LABEL_COLUMNS].sort_values("nct_id")
        path = out_dir / labels_filename(phase)
        _write_bytes_atomic(path, rows.to_csv(index=False).encode("utf-8"))
        paths[phase] = path
    ids = sorted(top["nct_id"].unique())
    _write_bytes_atomic(out_dir / "nct_ids.txt", ("\n".join(ids) + "\n").encode("utf-8"))
    return paths


def load_splits(labels_path: str | Path) -> pd.Series:
    """Return each trial's TOP split (``train``, ``valid`` or ``test``), indexed by ``nct_id``.

    Join it to a ``dataset.parquet`` to separate training and test rows; keep it out of the
    model's features.
    """
    frame = pd.read_csv(labels_path, usecols=["nct_id", "split"], dtype="string")
    validate_nct_ids(frame["nct_id"], source=str(labels_path))
    unknown = sorted(set(frame["split"].dropna()) - set(SPLITS))
    if frame["split"].isna().any() or unknown:
        raise ValueError(f"{labels_path}: split must be one of {SPLITS}, found {unknown or 'NA'}")
    return frame.set_index("nct_id")["split"].sort_index()


def build_top_datasets(
    raw_path: str | Path,
    output_root: str | Path,
    labels_dir: str | Path = LABELS_DIR,
) -> dict[str, dict[str, Any]]:
    """Build one labeled dataset per phase from one registry snapshot; return the manifests.

    Output goes to ``<output_root>/phase_<P>/``. Nothing is built if any of them exists.
    """
    output_root, labels_dir = Path(output_root), Path(labels_dir)
    outputs = {phase: output_root / f"phase_{phase}" for phase in PHASES}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"output directories already exist: {existing}; choose a new root")
    return {
        phase: build_dataset(
            raw_path,
            output,
            labels_path=labels_dir / labels_filename(phase),
            label_description=LABEL_DESCRIPTION.format(phase=phase),
        )
        for phase, output in outputs.items()
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
            temporary = Path(f.name)
            f.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _summarize(top: pd.DataFrame) -> dict[str, Any]:
    summary = {}
    for phase, rows in top.groupby("phase", sort=False):
        summary[f"phase_{phase}"] = {
            split: {"trials": len(group), "success_rate": round(float(group["label"].mean()), 3)}
            for split, group in rows.groupby("split", sort=False)
        }
    summary["unique_trials"] = int(top["nct_id"].nunique())
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TOP benchmark labels and per-phase datasets.")
    commands = parser.add_subparsers(dest="command", required=True)
    labels = commands.add_parser("labels", help="download TOP and write per-phase labels files")
    labels.add_argument("--top-dir", type=Path, default=TOP_DIR)
    labels.add_argument("--out-dir", type=Path, default=LABELS_DIR)
    datasets = commands.add_parser("datasets", help="build one labeled dataset per phase")
    datasets.add_argument("--raw", required=True, type=Path, help="registry snapshot (.jsonl)")
    datasets.add_argument("--output-root", type=Path, default=PROCESSED_DATA_DIR / "top")
    datasets.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    args = parser.parse_args(argv)

    try:
        if args.command == "labels":
            download_top(args.top_dir)
            top = load_top(args.top_dir)
            write_label_files(top, args.out_dir)
            result = {"labels_dir": str(args.out_dir), **_summarize(top)}
        else:
            manifests = build_top_datasets(args.raw, args.output_root, args.labels_dir)
            result = {
                "output_root": str(args.output_root),
                **{f"phase_{p}": m["counts"] for p, m in manifests.items()},
            }
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
