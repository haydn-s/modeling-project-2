from typing import Any

import pytest

from trial_outcome.data import ctgov


class FakeResponse:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    """Serves canned pages keyed by pageToken and records the params of each request."""

    def __init__(self, pages: dict[str | None, dict[str, Any]]):
        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], timeout: float) -> FakeResponse:
        assert url == ctgov.BASE_URL
        self.calls.append(dict(params))
        return FakeResponse(self.pages[params.get("pageToken")])


def study(nct_id: str) -> dict[str, Any]:
    return {"protocolSection": {"identificationModule": {"nctId": nct_id}}}


@pytest.fixture
def two_page_session() -> FakeSession:
    return FakeSession(
        {
            None: {"studies": [study("NCT1"), study("NCT2")], "nextPageToken": "page2"},
            "page2": {"studies": [study("NCT3")]},
        }
    )


def test_iter_studies_follows_pagination(two_page_session):
    studies = list(ctgov.iter_studies({"query.cond": "asthma"}, session=two_page_session))

    assert [s["protocolSection"]["identificationModule"]["nctId"] for s in studies] == [
        "NCT1",
        "NCT2",
        "NCT3",
    ]
    assert len(two_page_session.calls) == 2
    assert two_page_session.calls[0]["query.cond"] == "asthma"
    assert "pageToken" not in two_page_session.calls[0]
    assert two_page_session.calls[1]["pageToken"] == "page2"


def test_iter_studies_respects_max_studies(two_page_session):
    studies = list(ctgov.iter_studies(session=two_page_session, max_studies=2))

    assert len(studies) == 2
    assert len(two_page_session.calls) == 1
    assert two_page_session.calls[0]["pageSize"] == 2


@pytest.mark.parametrize(
    "kwargs", [{"page_size": 0}, {"page_size": ctgov.MAX_PAGE_SIZE + 1}, {"timeout": 0}]
)
def test_iter_studies_rejects_bad_arguments_on_call(kwargs):
    # No list(): the error must come from the call itself, not from the first iteration.
    with pytest.raises(ValueError):
        ctgov.iter_studies(session=FakeSession({}), **kwargs)


def test_download_round_trips_through_jsonl(tmp_path, two_page_session):
    out = tmp_path / "raw" / "studies.jsonl"

    n = ctgov.download_studies(out, session=two_page_session, progress=False)

    assert n == 3
    assert ctgov.read_jsonl(out) == [study("NCT1"), study("NCT2"), study("NCT3")]


@pytest.mark.parametrize("raw", ["no-equals-sign", "=value"])
def test_parse_param_rejects_malformed(raw):
    with pytest.raises(Exception, match="KEY=VALUE"):
        ctgov._parse_param(raw)


def test_parse_param_keeps_equals_in_value():
    assert ctgov._parse_param("filter.advanced=AREA[Phase]PHASE2") == (
        "filter.advanced",
        "AREA[Phase]PHASE2",
    )


def test_main_count_only_needs_no_out(monkeypatch, capsys):
    seen = {}

    def fake_count(params):
        seen["params"] = params
        return 42

    monkeypatch.setattr(ctgov, "count_studies", fake_count)

    ctgov.main(["--count-only", "--param", "query.cond=asthma"])

    assert capsys.readouterr().out.strip() == "42"
    assert seen["params"] == {"query.cond": "asthma"}


def test_main_requires_out_when_downloading(capsys):
    with pytest.raises(SystemExit) as excinfo:
        ctgov.main(["--param", "query.cond=asthma"])

    assert excinfo.value.code == 2
    assert "--out is required" in capsys.readouterr().err


@pytest.mark.parametrize("limit", [0, -1])
def test_iter_studies_rejects_nonpositive_limits(limit):
    session = FakeSession({})
    with pytest.raises(ValueError, match="max_studies must be positive"):
        ctgov.iter_studies(max_studies=limit, session=session)
    assert session.calls == []


def test_failed_download_preserves_existing_snapshot(tmp_path):
    path = tmp_path / "studies.jsonl"
    path.write_text('{"previous": true}\n')
    session = FakeSession({None: {"studies": [study("NCT1")], "nextPageToken": "missing"}})
    with pytest.raises(KeyError):
        ctgov.download_studies(path, session=session, progress=False)
    assert path.read_text() == '{"previous": true}\n'
    assert list(tmp_path.iterdir()) == [path]


def test_owned_sessions_are_closed(monkeypatch):
    session = FakeSession({None: {"studies": [], "totalCount": 0}})
    closed = []
    session.close = lambda: closed.append(True)
    monkeypatch.setattr(ctgov, "make_session", lambda: session)
    assert list(ctgov.iter_studies()) == []
    assert ctgov.count_studies() == 0
    assert closed == [True, True]


def test_repeated_pagination_token_fails():
    session = FakeSession(
        {
            None: {"studies": [], "nextPageToken": "same"},
            "same": {"studies": [], "nextPageToken": "same"},
        }
    )
    with pytest.raises(ValueError, match="repeated nextPageToken"):
        list(ctgov.iter_studies(session=session))


@pytest.mark.parametrize("payload", [{}, {"studies": {}}, {"studies": [42]}])
def test_malformed_api_response_fails(payload):
    with pytest.raises(ValueError, match="studies list"):
        list(ctgov.iter_studies(session=FakeSession({None: payload})))


@pytest.mark.parametrize("content", ["{}\nnot-json\n", "{}\n[1, 2]\n"])
def test_jsonl_reports_path_and_line(tmp_path, content):
    path = tmp_path / "malformed.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError, match="malformed.jsonl:2"):
        ctgov.read_jsonl(path)


class IdFilterSession:
    """Serves the records whose IDs appear in ``filter.ids``, in one page per request."""

    def __init__(self, available: list[str], extra: list[str] | None = None):
        self.available = available
        self.extra = extra or []
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], timeout: float) -> FakeResponse:
        self.calls.append(dict(params))
        requested = params["filter.ids"].split(",")
        found = [trial_id for trial_id in requested if trial_id in self.available]
        return FakeResponse({"studies": [study(i) for i in found + self.extra]})

    def close(self) -> None:
        pass


def ids(*numbers: int) -> list[str]:
    return [f"NCT{n:08d}" for n in numbers]


def test_iter_studies_by_id_batches_and_dedupes():
    session = IdFilterSession(ids(1, 2, 3, 4, 5))
    studies = ctgov.iter_studies_by_id(
        ids(1, 2, 2, 3, 4, 5), {"fields": "protocolSection"}, batch_size=2, session=session
    )
    assert [ctgov.study_nct_id(s) for s in studies] == ids(1, 2, 3, 4, 5)
    assert [call["filter.ids"] for call in session.calls] == [
        "NCT00000001,NCT00000002",
        "NCT00000003,NCT00000004",
        "NCT00000005",
    ]
    assert [call["pageSize"] for call in session.calls] == [2, 2, 1]
    assert all(call["fields"] == "protocolSection" for call in session.calls)


def test_download_by_id_reports_missing_trials(tmp_path):
    out = tmp_path / "raw" / "top.jsonl"
    session = IdFilterSession(ids(1, 3))
    missing = ctgov.download_studies_by_id(out, ids(1, 2, 3), session=session, progress=False)
    assert missing == ids(2)
    assert ctgov.read_jsonl(out) == [study(i) for i in ids(1, 3)]


@pytest.mark.parametrize("extra,message", [(ids(9), "not requested"), (ids(1), "more than once")])
def test_ignored_id_filter_fails_without_writing(tmp_path, extra, message):
    out = tmp_path / "top.jsonl"
    session = IdFilterSession(ids(1, 2), extra=extra)
    with pytest.raises(ValueError, match=message):
        ctgov.download_studies_by_id(out, ids(1, 2), session=session, progress=False)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "requested,kwargs,message",
    [
        ([], {}, "no NCT IDs"),
        (["NCT123"], {}, "invalid nct_id"),
        (ids(1), {"batch_size": 0}, "batch_size"),
        (ids(1), {"params": {"filter.ids": "NCT00000001"}}, "not as a filter.ids"),
    ],
)
def test_iter_studies_by_id_rejects_bad_requests(requested, kwargs, message):
    session = IdFilterSession([])
    with pytest.raises(ValueError, match=message):
        list(ctgov.iter_studies_by_id(requested, session=session, **kwargs))
    assert session.calls == []


def test_read_id_file_skips_blanks_and_comments(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("# TOP trials\nNCT00000001\n\n  NCT00000002  \n")
    assert ctgov.read_id_file(path) == ids(1, 2)


def test_cli_downloads_ids_and_rejects_conflicting_options(tmp_path, monkeypatch, capsys):
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("NCT00000001\nNCT00000002\n")
    out = tmp_path / "top.jsonl"
    monkeypatch.setattr(ctgov, "make_session", lambda: IdFilterSession(ids(1)))
    monkeypatch.setattr(ctgov, "tqdm", lambda studies, **kwargs: studies)

    ctgov.main(["--ids-file", str(ids_file), "--out", str(out)])

    output = capsys.readouterr().out
    assert "Wrote 1 studies" in output
    assert "1 requested IDs were not returned" in output
    with pytest.raises(SystemExit):
        ctgov.main(["--ids-file", str(ids_file), "--out", str(out), "--count-only"])
