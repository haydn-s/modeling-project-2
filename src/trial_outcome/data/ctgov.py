"""Client for the ClinicalTrials.gov API v2.

API reference: https://clinicaltrials.gov/data-api/api

Query parameters are passed through unchanged, so anything the API supports works, e.g.
``query.cond``, ``query.intr``, ``query.term``, ``filter.overallStatus``, ``filter.advanced``,
and ``fields``.

Command-line usage (writes one JSON study record per line)::

    python -m trial_outcome.data.ctgov \\
        --param query.cond="breast cancer" \\
        --param "filter.overallStatus=COMPLETED|TERMINATED" \\
        --max-studies 5000 \\
        --out data/raw/ctgov_breast_cancer.jsonl

To fetch a known set of trials, list one NCT ID per line in a text file::

    python -m trial_outcome.data.ctgov --ids-file data/interim/top_labels/nct_ids.txt \\
        --out data/raw/ctgov_top.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing, nullcontext
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from trial_outcome.data.validation import validate_nct_ids

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
MAX_PAGE_SIZE = 1000
# 200 IDs make a ~2.4 KB query string, well under common URL length limits.
MAX_IDS_PER_REQUEST = 200


def make_session(retries: int = 5, backoff_factor: float = 1.0) -> requests.Session:
    """Return a session that retries rate-limited (429) and transient server errors."""
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def iter_studies(
    params: Mapping[str, Any] | None = None,
    *,
    page_size: int = MAX_PAGE_SIZE,
    max_studies: int | None = None,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> Iterator[dict[str, Any]]:
    """Yield study records matching ``params``, following ``nextPageToken`` pagination.

    Args:
        params: API query parameters, e.g. ``{"query.cond": "asthma"}``.
        page_size: Studies per request (1-1000).
        max_studies: Stop after this many studies (``None`` fetches everything).
        session: Optional ``requests.Session``; defaults to :func:`make_session`.
        timeout: Per-request timeout in seconds.

    Raises:
        ValueError: If an argument is out of range. Raised by the call itself, before any
            request is made, rather than on the first iteration.
    """
    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}, got {page_size}")
    if max_studies is not None:
        if max_studies < 1:
            raise ValueError("max_studies must be positive")
        page_size = min(page_size, max_studies)
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    query: dict[str, Any] = {**(params or {}), "pageSize": page_size, "format": "json"}
    return _paginate(query, max_studies=max_studies, session=session, timeout=timeout)


def _paginate(
    query: dict[str, Any],
    *,
    max_studies: int | None,
    session: requests.Session | None,
    timeout: float,
) -> Iterator[dict[str, Any]]:
    """Yield studies page by page; ``iter_studies`` validates the arguments first."""
    context = nullcontext(session) if session is not None else closing(make_session())
    with context as client:
        yielded = 0
        seen_tokens = set()
        while True:
            response = client.get(BASE_URL, params=query, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            studies = payload.get("studies")
            if not isinstance(studies, list) or any(not isinstance(s, dict) for s in studies):
                raise ValueError("API response must contain a studies list of objects")

            for study in studies:
                yield study
                yielded += 1
                if max_studies is not None and yielded >= max_studies:
                    return

            next_token = payload.get("nextPageToken")
            if not next_token:
                return
            if next_token in seen_tokens:
                raise ValueError("API returned a repeated nextPageToken")
            seen_tokens.add(next_token)
            query["pageToken"] = next_token


def study_nct_id(study: Mapping[str, Any]) -> Any:
    """Return a study record's NCT ID, or ``None`` if the record has none."""
    return study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")


def iter_studies_by_id(
    nct_ids: Iterable[str],
    params: Mapping[str, Any] | None = None,
    *,
    batch_size: int = MAX_IDS_PER_REQUEST,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> Iterator[dict[str, Any]]:
    """Yield the records for the given NCT IDs, requesting ``batch_size`` IDs at a time.

    Duplicate IDs are requested once. IDs the registry does not return are skipped; callers
    can compare the yielded IDs with the request. A record whose ID was not requested, or
    that arrives twice, raises ``ValueError`` because it means the ID filter was ignored.
    Other ``params`` (e.g. ``fields``) are sent with every batch.
    """
    ids = list(dict.fromkeys(nct_ids))
    if not ids:
        raise ValueError("no NCT IDs requested")
    validate_nct_ids(pd.Index(ids, dtype=object), source="requested IDs")
    if not 1 <= batch_size <= MAX_PAGE_SIZE:
        raise ValueError(f"batch_size must be between 1 and {MAX_PAGE_SIZE}, got {batch_size}")
    if params and "filter.ids" in params:
        raise ValueError("pass trial IDs as nct_ids, not as a filter.ids parameter")

    requested, seen = set(ids), set()
    context = nullcontext(session) if session is not None else closing(make_session())
    with context as client:
        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            query = {**(params or {}), "filter.ids": ",".join(batch)}
            for study in iter_studies(query, page_size=len(batch), session=client, timeout=timeout):
                trial_id = study_nct_id(study)
                if trial_id not in requested:
                    raise ValueError(
                        f"API returned {trial_id!r}, which was not requested; "
                        "the filter.ids parameter was not applied"
                    )
                if trial_id in seen:
                    raise ValueError(f"API returned {trial_id} more than once")
                seen.add(trial_id)
                yield study


def count_studies(
    params: Mapping[str, Any] | None = None, *, session: requests.Session | None = None
) -> int:
    """Return the total number of studies matching ``params`` without downloading them."""
    query = {**(params or {}), "pageSize": 1, "countTotal": "true", "format": "json"}
    context = nullcontext(session) if session is not None else closing(make_session())
    with context as client:
        response = client.get(BASE_URL, params=query, timeout=60)
        response.raise_for_status()
        return int(response.json()["totalCount"])


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    """Atomically write JSON Lines; leave any previous file intact on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as f:
            temporary = Path(f.name)
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return n


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read study objects with actionable filename/line errors for malformed input."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a study object")
            yield record


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON Lines file written by :func:`write_jsonl`."""
    return list(iter_jsonl(path))


def download_studies(
    path: str | Path,
    params: Mapping[str, Any] | None = None,
    *,
    max_studies: int | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> int:
    """Download all studies matching ``params`` to a JSON Lines file. Returns the count."""
    studies = iter_studies(params, max_studies=max_studies, **kwargs)
    if progress:
        studies = tqdm(studies, total=max_studies, unit="study", desc="ClinicalTrials.gov")
    return write_jsonl(studies, path)


def download_studies_by_id(
    path: str | Path,
    nct_ids: Iterable[str],
    params: Mapping[str, Any] | None = None,
    *,
    progress: bool = True,
    **kwargs: Any,
) -> list[str]:
    """Download the listed trials to a JSON Lines file. Returns the IDs that were not found."""
    ids = list(dict.fromkeys(nct_ids))
    found: set[str] = set()

    def tracked() -> Iterator[dict[str, Any]]:
        for study in iter_studies_by_id(ids, params, **kwargs):
            found.add(study_nct_id(study))
            yield study

    studies: Iterable[dict[str, Any]] = tracked()
    if progress:
        studies = tqdm(studies, total=len(ids), unit="study", desc="ClinicalTrials.gov")
    write_jsonl(studies, path)
    return [trial_id for trial_id in ids if trial_id not in found]


def read_id_file(path: str | Path) -> list[str]:
    """Read NCT IDs, one per line; blank lines and ``#`` comments are ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def _parse_param(raw: str) -> tuple[str, str]:
    key, sep, value = raw.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {raw!r}")
    return key, value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download studies from ClinicalTrials.gov.")
    parser.add_argument("--out", type=Path, help="output .jsonl path (unused with --count-only)")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        type=_parse_param,
        metavar="KEY=VALUE",
        help="API query parameter (repeatable), e.g. query.cond=asthma",
    )
    parser.add_argument("--max-studies", type=int, default=None)
    parser.add_argument("--count-only", action="store_true", help="print the match count and exit")
    parser.add_argument(
        "--ids-file", type=Path, help="download exactly these trials (one NCT ID per line)"
    )
    args = parser.parse_args(argv)

    params = dict(args.param)
    if args.ids_file is not None:
        if args.count_only or args.max_studies is not None:
            parser.error("--ids-file cannot be combined with --count-only or --max-studies")
        try:
            ids = read_id_file(args.ids_file)
            missing = download_studies_by_id(args.out, ids, params)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        found = len(set(ids)) - len(missing)
        print(f"Wrote {found} studies to {args.out}")
        if missing:
            print(f"{len(missing)} requested IDs were not returned, e.g. {missing[:5]}")
        return
    if args.count_only:
        print(count_studies(params))
        return
    if args.out is None:
        parser.error("--out is required unless --count-only is given")
    n = download_studies(args.out, params, max_studies=args.max_studies)
    print(f"Wrote {n} studies to {args.out}")


if __name__ == "__main__":
    main()
