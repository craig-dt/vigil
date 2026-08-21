"""Guard that the compose data plane is not published on 0.0.0.0.

Issue #587: `infra/docker/docker-compose.yml` published Postgres, Redis and
Bifrost with bare `"5432:5432"`-style mappings. Docker reads a two-part mapping
as `0.0.0.0:<host>:<container>`, so every one of them was reachable from any
host on the operator's network -- holding findings, case data, the ARQ queue and
LLM provider credentials respectively.

None of those has a consumer outside the host: the backend, daemon and workers
reach them over the `deeptempo-network` bridge by service name, and a host-run
backend (`start.sh`) reaches them over loopback. So the host-side interface can
be pinned to 127.0.0.1 without breaking anything.

This is a gate rather than a one-time fix. The original exposure was invisible
in review precisely because a bare `"5432:5432"` looks deliberate, so the next
service added to the data plane would repeat it silently.

`SHARED_HOST_SERVICES` is the deliberate opposite list: services that really do
have off-host consumers (analyst browsers, SIEM push, metrics scraping) and are
therefore *expected* to publish broadly. Adding a name to it is the explicit
act of saying "this one is meant to be reachable from the network".
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.config import REPO_ROOT

pytestmark = pytest.mark.unit

COMPOSE_PATH = Path("infra/docker/docker-compose.yml")

# Services whose whole point is off-host reachability. Everything else in the
# default (profile-less) set must bind loopback.
SHARED_HOST_SERVICES = frozenset(
    {
        "backend",  # analyst browsers
        "soc-daemon",  # SIEM webhook push + Prometheus scraping
    }
)

# Services whose loopback bind is already in flight in another PR, so this branch
# deliberately leaves the line alone rather than collide on it.
#
# Held as a *strict* xfail rather than an allowlist entry: once that PR lands and
# the service binds loopback, this test starts XPASSing and CI goes red until the
# entry below is deleted. The exception therefore cannot outlive its reason, which
# a plain allowlist entry silently would.
PENDING_ELSEWHERE = {
    "bifrost": "loopback bind ships with PR #703 (Bifrost config via Vigil settings)",
}


def _services() -> dict:
    raw = yaml.safe_load((REPO_ROOT / COMPOSE_PATH).read_text(encoding="utf-8"))
    return raw.get("services", {}) or {}


def _host_interface(mapping) -> str:
    """The interface a compose `ports:` entry publishes on.

    Every entry under `ports:` publishes something, so this always names an
    interface; only an explicit prefix narrows it. Compose accepts the short
    string form and the long dict form. In the short form Docker treats a
    two-part `"5432:5432"` as 0.0.0.0 and a three-part `"127.0.0.1:5432:5432"`
    as loopback; a bare `"5432"` also lands on 0.0.0.0, just on an ephemeral
    host port. The long form carries the interface in `host_ip`, which likewise
    defaults to 0.0.0.0 when absent.
    """
    if isinstance(mapping, dict):
        return str(mapping.get("host_ip") or "0.0.0.0")

    parts = str(mapping).rsplit(":", 2)
    return parts[0] if len(parts) == 3 else "0.0.0.0"


def _published(spec: dict) -> list[tuple[str, str]]:
    return [(str(m), _host_interface(m)) for m in (spec.get("ports") or [])]


def _default_profile_services() -> dict:
    """Only services started by a bare `docker compose up`.

    Anything behind a `profiles:` key (pgadmin, the observability stack, splunk,
    kafka) is opt-in developer tooling and out of scope here.
    """
    return {
        name: spec
        for name, spec in _services().items()
        if isinstance(spec, dict) and not spec.get("profiles")
    }


def test_compose_file_is_where_this_test_thinks_it_is():
    # Guards the rest of the file against silently passing on an empty parse if
    # the compose file is moved again -- it already moved once (docker/ ->
    # infra/docker/) after #587 was filed.
    assert _default_profile_services(), f"no profile-less services parsed from {COMPOSE_PATH}"


def _service_params() -> list:
    params = []
    for name in sorted(_default_profile_services()):
        marks = (
            [pytest.mark.xfail(strict=True, reason=PENDING_ELSEWHERE[name])]
            if name in PENDING_ELSEWHERE
            else []
        )
        params.append(pytest.param(name, marks=marks))
    return params


@pytest.mark.parametrize("service", _service_params())
def test_data_plane_publishes_on_loopback_only(service):
    spec = _default_profile_services()[service]
    if service in SHARED_HOST_SERVICES:
        pytest.skip(f"{service} is a declared off-host service")

    wide = [
        mapping
        for mapping, iface in _published(spec)
        if iface not in {"127.0.0.1", "::1", "localhost"}
    ]
    assert not wide, (
        f"{service} publishes {wide} on a non-loopback interface in {COMPOSE_PATH}. "
        "Prefix the mapping with 127.0.0.1: , or add the service to "
        "SHARED_HOST_SERVICES if it genuinely needs off-host consumers (#587)."
    )


def test_pending_elsewhere_services_are_real_services():
    # Same anti-rot check as below: an entry naming a service that no longer
    # exists would never XPASS, so it would sit here forever unnoticed.
    unknown = set(PENDING_ELSEWHERE) - set(_services())
    assert not unknown, f"PENDING_ELSEWHERE names services not in compose: {sorted(unknown)}"


def test_shared_host_services_are_real_services():
    # Keeps the allowlist from rotting into a list of names that no longer exist,
    # which would silently stop gating anything.
    unknown = SHARED_HOST_SERVICES - set(_services())
    assert not unknown, f"SHARED_HOST_SERVICES names services not in compose: {sorted(unknown)}"
