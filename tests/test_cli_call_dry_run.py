"""`treg call --dry-run`, `treg call --format`, and the catalog search filters.

The dry-run quote reads the same PUBLIC detail route `catalog get` reads, so these tests run the
CLI handler against a fake client — no server, no upstream, no money. The formatter tests pin the
one contract agents parse: a JSON list of objects becomes a markdown table, and an error body
NEVER gets reformatted.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from treg import cli


# ---- fakes -------------------------------------------------------------------------------------

class FakeClient:
    """Records requests; answers /catalog/endpoints/... from a dict of bodies."""

    def __init__(self, bodies: dict[str, object] | None = None, status: int = 200):
        self.bodies = bodies or {}
        self.status = status
        self.requests: list[tuple[str, str]] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, path, params=None):
        self.requests.append(("GET", path))
        if path in self.bodies:
            body = self.bodies[path]
        else:
            body = _detail_body()
        return httpx.Response(self.status, json=body, request=httpx.Request("GET", f"http://t{path}"))

    def request(self, method, url, params=None, content=None, headers=None):
        self.requests.append((method, str(url)))
        return httpx.Response(200, json={}, request=httpx.Request(method, f"http://t{url}"))


def _detail_body(**overrides) -> dict:
    """A minimal /catalog/endpoints/<id> body shaped like the real route's answer."""
    ep = {
        "id": "acme.web.search",
        "provider": "acme",
        "provider_display": "Acme",
        "summary": "Search the web",
        "method": "POST",
        "path": "/v1/search",
        "kind": "data",
        "cost": {"value": 0.5, "currency": "USD", "type": "per_call", "usd": 0.5},
        "input": {
            "queryParams": {"q": {"type": "string", "required": True, "example": "hello"}},
            "body": {"limit": {"type": "integer", "required": False, "default": 10}},
        },
        "verified": "2026-09-21",
        "tier": "core",
    }
    ep.update(overrides)
    return {"endpoint": ep, "provider": {"service": "acme", "display_name": "Acme"},
            "call_template": "treg call acme.web.search", "example_response": {"results": [{"title": "a"}]}}


def _args(**kw) -> SimpleNamespace:
    base = dict(target="acme.web.search", path="", method=None, query=[], data=None, file=None,
                content_type=None, header=[], upload=[], await_task=False, dry_run=True, format=None)
    base.update(kw)
    return SimpleNamespace(**base)


# ---- dry-run -----------------------------------------------------------------------------------

def test_dry_run_quotes_cost_and_never_sends(monkeypatch, capsys):
    client = FakeClient()
    monkeypatch.setattr(cli, "_client", lambda cfg, **k: client)
    cli.cmd_call(_args(), {})
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "$0.5/call" in out or "$0.5" in out
    assert "billed to" in out
    # The one load-bearing assertion: the proxy call path is never reached.
    assert all(not path.startswith("/call/") for _m, path in client.requests)


def test_dry_run_flags_missing_required_params(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_client", lambda cfg, **k: FakeClient())
    cli.cmd_call(_args(), {})  # no --query sent
    out = capsys.readouterr().out
    assert "missing required params" in out and "q" in out


def test_dry_run_marks_sent_params_with_check(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_client", lambda cfg, **k: FakeClient())
    cli.cmd_call(_args(query=["q=hello"], data='{"limit": 5}'), {})
    out = capsys.readouterr().out
    assert "✓" in out
    assert "all required params present" in out


def test_dry_run_free_endpoint_says_free(monkeypatch, capsys):
    body = _detail_body(cost={"value": 0, "currency": "USD", "type": "per_call", "usd": 0})
    monkeypatch.setattr(cli, "_client", lambda cfg, **k: FakeClient({"/catalog/endpoints/endpoint-free": body}))
    cli.cmd_call(_args(target="endpoint-free"), {})
    out = capsys.readouterr().out
    assert "free" in out


def test_dry_run_unknown_price_says_refused(monkeypatch, capsys):
    body = _detail_body(cost={"value": 2, "currency": "CRED", "type": "per_call"})
    monkeypatch.setattr(cli, "_client", lambda cfg, **k: FakeClient({"/catalog/endpoints/endpoint-unpriced": body}))
    cli.cmd_call(_args(target="endpoint-unpriced"), {})
    out = capsys.readouterr().out
    assert "unknown" in out and "connections connect" in out


def test_dry_run_shows_example_response(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_client", lambda cfg, **k: FakeClient())
    cli.cmd_call(_args(), {})
    out = capsys.readouterr().out
    assert "example response" in out and '"title": "a"' in out


def test_dry_run_raw_url_is_refused(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        cli.cmd_call(_args(target="https://api.acme.com/v1/search"), {})


def test_dry_run_unknown_endpoint_falls_through_to_404(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_client", lambda cfg, **k: FakeClient(status=404))
    with pytest.raises(SystemExit):
        cli.cmd_call(_args(target="nope.nope.nope"), {})


# ---- --format -----------------------------------------------------------------------------------

def _call_args(fmt: str | None) -> SimpleNamespace:
    return _args(dry_run=False, format=fmt, await_task=False)


def test_format_markdown_renders_list_as_table(monkeypatch, capsys):
    seen = []

    class BodyClient(FakeClient):
        def request(self, method, url, params=None, content=None, headers=None):
            seen.append(url)
            return httpx.Response(200, json={"data": [{"name": "a", "id": 1}, {"name": "b", "id": 2}]},
                                  request=httpx.Request(method, f"http://t{url}"))

    monkeypatch.setattr(cli, "_client", lambda cfg, **k: BodyClient())
    cli.cmd_call(_call_args("markdown"), {})
    out = capsys.readouterr().out
    assert "| name | id |" in out.replace(" ", " ") or "| name |" in out
    assert "| a | 1 |" in out
    assert "| b | 2 |" in out


def test_format_bare_list_wraps_in_items(monkeypatch, capsys):
    class BodyClient(FakeClient):
        def request(self, method, url, params=None, content=None, headers=None):
            return httpx.Response(200, json=[{"k": "v"}], request=httpx.Request(method, f"http://t{url}"))

    monkeypatch.setattr(cli, "_client", lambda cfg, **k: BodyClient())
    cli.cmd_call(_call_args("markdown"), {})
    out = capsys.readouterr().out
    assert "| k |" in out and "| v |" in out


def test_format_json_stays_indented(monkeypatch, capsys):
    class BodyClient(FakeClient):
        def request(self, method, url, params=None, content=None, headers=None):
            return httpx.Response(200, json={"data": [{"a": 1}]}, request=httpx.Request(method, f"http://t{url}"))

    monkeypatch.setattr(cli, "_client", lambda cfg, **k: BodyClient())
    cli.cmd_call(_call_args("json"), {})
    out = capsys.readouterr().out
    assert json.loads(out)["data"][0]["a"] == 1


def test_format_never_touches_an_error_body(monkeypatch, capsys):
    class BodyClient(FakeClient):
        def request(self, method, url, params=None, content=None, headers=None):
            return httpx.Response(402, json={"detail": {"balance_micro": 0}},
                                  request=httpx.Request(method, f"http://t{url}"))

    monkeypatch.setattr(cli, "_client", lambda cfg, **k: BodyClient())
    with pytest.raises(SystemExit):
        cli.cmd_call(_call_args("markdown"), {})
    out = capsys.readouterr().out
    assert "balance_micro" in out          # verbatim JSON, not a table
    assert "| " not in out


def test_format_passes_non_json_through(monkeypatch, capsys):
    class BodyClient(FakeClient):
        def request(self, method, url, params=None, content=None, headers=None):
            return httpx.Response(200, text="plain text answer",
                                  headers={"content-type": "text/plain"},
                                  request=httpx.Request(method, f"http://t{url}"))

    monkeypatch.setattr(cli, "_client", lambda cfg, **k: BodyClient())
    cli.cmd_call(_call_args("markdown"), {})
    assert "plain text answer" in capsys.readouterr().out


# ---- catalog search filters ----------------------------------------------------------------------

SEARCH_BODY = {
    "query": "video", "count": 3, "total": 3,
    "results": [
        {"id": "a.video.search", "provider": "a", "summary": "s", "method": "GET",
         "capability": "video.search", "domain": "video", "platform": "web",
         "kind": "data", "verified": True, "tier": "core",
         "cost": {"value": 0.001, "currency": "USD", "type": "per_call", "usd": 0.001}},
        {"id": "b.video.upscale", "provider": "b", "summary": "s", "method": "POST",
         "capability": "video.upscale", "domain": "video", "platform": "web",
         "kind": "data", "verified": True, "tier": "core",
         "cost": {"value": 0.5, "currency": "USD", "type": "per_call", "usd": 0.5}},
        {"id": "c.backlinks.list", "provider": "c", "summary": "s", "method": "GET",
         "capability": "backlinks.list", "domain": "backlinks", "platform": "web",
         "kind": "data", "verified": True, "tier": "core",
         "cost": {"value": 0.02, "currency": "USD", "type": "per_call", "usd": 0.02}},
    ],
    "hints": [],
}


@pytest.fixture(autouse=True)
def _reset_json_override(monkeypatch):
    # test_cli.py's host --json test leaves the module-global on in this shared process; every
    # direct cmd_* call here must see the default. (Same convention as test_org_rename.)
    monkeypatch.setattr(cli, "_JSON_OVERRIDE", False)


def _search_args(**kw) -> SimpleNamespace:
    base = dict(platform="search", rest=["video"], limit=25, under=None, category=None, show_all=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_search_under_filter_drops_pricier_rows(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_client", lambda cfg, **k: FakeClient())
    monkeypatch.setattr(cli, "_JSON_OVERRIDE", False)
    import treg.cli as c
    orig_get = FakeClient.get

    def patched(self, path, params=None):
        return httpx.Response(200, json=SEARCH_BODY, request=httpx.Request("GET", f"http://t{path}"))

    monkeypatch.setattr(cli, "_client", lambda cfg, **k: type("C", (), {
        "__enter__": lambda s: s, "__exit__": lambda s, *a: False, "get": patched})())
    cli._catalog_search("video", _search_args(under=0.1), {})
    out = capsys.readouterr().out
    assert "a.video.search" in out
    assert "b.video.upscale" not in out        # $0.50 >= $0.10
    assert "dropped" in out                    # never silent


def test_search_category_filter_keeps_domain_rows(monkeypatch, capsys):
    def patched(self, path, params=None):
        return httpx.Response(200, json=SEARCH_BODY, request=httpx.Request("GET", f"http://t{path}"))

    monkeypatch.setattr(cli, "_client", lambda cfg, **k: type("C", (), {
        "__enter__": lambda s: s, "__exit__": lambda s, *a: False, "get": patched})())
    cli._catalog_search("video", _search_args(category="back"), {})
    out = capsys.readouterr().out
    assert "c.backlinks.list" in out
    assert "a.video.search" not in out


def test_search_unknown_category_says_so(monkeypatch, capsys):
    def patched(self, path, params=None):
        return httpx.Response(200, json=SEARCH_BODY, request=httpx.Request("GET", f"http://t{path}"))

    monkeypatch.setattr(cli, "_client", lambda cfg, **k: type("C", (), {
        "__enter__": lambda s: s, "__exit__": lambda s, *a: False, "get": patched})())
    cli._catalog_search("video", _search_args(category="zzz"), {})
    out = capsys.readouterr().out
    assert "no rows in category 'zzz'" in out


def test_parser_accepts_new_flags():
    p = cli.build_parser()
    a = p.parse_args(["call", "x.y.z", "--dry-run", "--format", "markdown"])
    assert a.dry_run is True and a.format == "markdown"
    b = p.parse_args(["catalog", "search", "video", "--under", "0.02", "--category", "video"])
    assert b.under == 0.02 and b.category == "video"
