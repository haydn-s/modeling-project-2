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


def test_iter_studies_rejects_bad_page_size():
    with pytest.raises(ValueError):
        list(ctgov.iter_studies(page_size=ctgov.MAX_PAGE_SIZE + 1, session=FakeSession({})))


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


@pytest.mark.parametrize("limit", [0, -1])
def test_iter_studies_rejects_nonpositive_limits(limit):
    session = FakeSession({})
    with pytest.raises(ValueError, match="max_studies must be positive"):
        list(ctgov.iter_studies(max_studies=limit, session=session))
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
