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
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing, nullcontext
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
MAX_PAGE_SIZE = 1000


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


def _parse_param(raw: str) -> tuple[str, str]:
    key, sep, value = raw.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {raw!r}")
    return key, value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download studies from ClinicalTrials.gov.")
    parser.add_argument("--out", required=True, type=Path, help="output .jsonl path")
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
    args = parser.parse_args(argv)

    params = dict(args.param)
    if args.count_only:
        print(count_studies(params))
        return
    n = download_studies(args.out, params, max_studies=args.max_studies)
    print(f"Wrote {n} studies to {args.out}")


if __name__ == "__main__":
    main()
