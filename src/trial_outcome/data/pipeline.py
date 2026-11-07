"""Rebuild a validated feature dataset from a local ClinicalTrials.gov JSONL snapshot.

    python -m trial_outcome.data.pipeline --raw data/raw/studies.jsonl \\
        --output-dir data/processed/baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

from trial_outcome.config import PROCESSED_DATA_DIR
from trial_outcome.data.ctgov import read_jsonl
from trial_outcome.features import build_features, get_path, nct_id
from trial_outcome.labeling import build_labels

STUDY_TYPES = ("INTERVENTIONAL", "OBSERVATIONAL", "ANY")
SCHEMA_VERSION = 1


def _file_info(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def build_dataset(
    raw_path: str | Path,
    output_dir: str | Path,
    *,
    labels_path: str | Path | None = None,
    label_description: str | None = None,
    study_type: str = "INTERVENTIONAL",
) -> dict[str, Any]:
    """Write Parquet tables and a provenance manifest into a new output directory.

    The default cohort is interventional studies. Unlabeled trials stay in the feature
    table; only matched labels enter ``dataset.parquet``. All exclusions and unmatched
    labels are reported. Existing outputs are never overwritten, and failed builds
    never publish a partial output directory. Returns the manifest.
    """
    raw_path, output_dir = Path(raw_path), Path(output_dir)
    if study_type not in STUDY_TYPES:
        raise ValueError(f"study_type must be one of {STUDY_TYPES}")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}; choose a new path")
    if labels_path is not None and (label_description is None or not label_description.strip()):
        raise ValueError(
            "label_description is required with labels: state the source and definition"
        )
    if labels_path is None and label_description is not None:
        raise ValueError("label_description requires a labels file")

    inputs = {"raw": _file_info(raw_path)}
    studies = read_jsonl(raw_path)
    if not studies:
        raise ValueError("raw input contains no studies")
    features = build_features(studies)
    included = (
        pd.Series(True, index=features.index)
        if study_type == "ANY"
        else features["study_type"].eq(study_type).fillna(False)
    )
    audit = (
        pd.DataFrame(
            [
                {
                    "nct_id": nct_id(study),
                    "overall_status": get_path(study, "protocolSection.statusModule.overallStatus"),
                    "first_posted_date": get_path(
                        study, "protocolSection.statusModule.studyFirstPostDateStruct.date"
                    ),
                    "last_update_date": get_path(
                        study, "protocolSection.statusModule.lastUpdatePostDateStruct.date"
                    ),
                }
                for study in studies
            ],
            dtype="string",
        )
        .set_index("nct_id")
        .sort_index()
    )
    audit["study_type"] = features["study_type"]
    audit["included"] = included
    audit["exclusion_reason"] = pd.Series("", index=audit.index, dtype="string")
    audit.loc[~included, "exclusion_reason"] = "study_type_mismatch_or_missing"
    features = features.loc[included]
    if features.empty:
        raise ValueError(f"no studies remain in the {study_type} cohort")

    tables = {"features.parquet": features, "cohort_audit.parquet": audit}
    counts = {
        "raw_studies": len(studies),
        "included_studies": len(features),
        "excluded_studies": int((~included).sum()),
    }
    if labels_path is not None:
        labels_path = Path(labels_path)
        inputs["labels"] = _file_info(labels_path)
        labels = build_labels(labels_path).set_index("nct_id")
        dataset = features.join(labels, how="inner", validate="one_to_one").sort_index()
        if dataset.empty:
            raise ValueError("no labels match the selected study cohort")
        unmatched = labels.loc[~labels.index.isin(features.index)].copy()
        unmatched["reason"] = [
            "excluded_from_cohort" if trial_id in audit.index else "not_in_raw"
            for trial_id in unmatched.index
        ]
        audit["has_label"] = audit.index.isin(labels.index)
        tables.update({"dataset.parquet": dataset, "unmatched_labels.parquet": unmatched})
        counts.update(
            labels_supplied=len(labels),
            labeled_studies=len(dataset),
            unlabeled_studies=len(features) - len(dataset),
            unmatched_labels=len(unmatched),
            success_labels=int(dataset["label"].eq(1).sum()),
            failure_labels=int(dataset["label"].eq(0).sum()),
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "inputs": inputs,
        "configuration": {"study_type": study_type, "label_description": label_description},
        "counts": counts,
        "feature_dtypes": {column: str(dtype) for column, dtype in features.dtypes.items()},
        "feature_missing_counts": {column: int(n) for column, n in features.isna().sum().items()},
        "limitations": [
            "Current registry records may contain later edits; not historical snapshots.",
            "No clinical success labels are inferred from registry completion status.",
            "Fit imputation, category encoding, and scaling on a training partition only.",
        ],
        "outputs": {},
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as temp:
        staging = Path(temp)
        for name, table in tables.items():
            path = staging / name
            table.to_parquet(path, index=True)
            info = _file_info(path)
            info["path"] = name
            info["rows"] = len(table)
            manifest["outputs"][name] = info
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # The staging directory is a sibling, so publication stays on one filesystem.
        if output_dir.exists():
            raise FileExistsError(f"output directory already exists: {output_dir}")
        staging.rename(output_dir)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path, help="local API v2 JSON Lines snapshot")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR / "baseline")
    parser.add_argument("--study-type", choices=STUDY_TYPES, default="INTERVENTIONAL")
    parser.add_argument("--labels", type=Path, help="CSV or Parquet with nct_id and binary label")
    parser.add_argument(
        "--label-description", help="label source and definition of success/failure"
    )
    args = parser.parse_args(argv)
    try:
        manifest = build_dataset(
            args.raw,
            args.output_dir,
            labels_path=args.labels,
            label_description=args.label_description,
            study_type=args.study_type,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"output_dir": str(args.output_dir), **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
