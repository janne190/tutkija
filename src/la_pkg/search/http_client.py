from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Mapping

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
USER_AGENT_ENV = "TUTKIJA_USER_AGENT"
CONTACT_ENV = "TUTKIJA_CONTACT_EMAIL"


@lru_cache(maxsize=None)
def get_contact_email() -> str | None:
    """Return the optional contact email used for polite API identification."""

    value = os.getenv(CONTACT_ENV, "").strip()
    return value or None


@lru_cache(maxsize=None)
def get_user_agent() -> str:
    """Build the default User-Agent header for outbound HTTP requests."""

    override = os.getenv(USER_AGENT_ENV, "").strip()
    if override:
        return override
    contact = get_contact_email()
    if contact:
        return f"Tutkija (+{contact})"
    return "Tutkija"


def _merge_headers(passed: Mapping[str, Any] | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if passed:
        headers.update({str(key): str(value) for key, value in passed.items()})
    if not any(key.lower() == "user-agent" for key in headers):
        headers["User-Agent"] = get_user_agent()
    return headers


def create_http_client(**kwargs: Any) -> httpx.Client:
    """Create an `httpx.Client` with shared defaults for Tutkija."""

    timeout = kwargs.pop("timeout", DEFAULT_TIMEOUT)
    follow_redirects = kwargs.pop("follow_redirects", True)
    headers_param = kwargs.pop("headers", None)
    headers = _merge_headers(
        headers_param if isinstance(headers_param, Mapping) else None
    )
    return httpx.Client(
        timeout=timeout,
        follow_redirects=follow_redirects,
        headers=headers,
        **kwargs,
    )


def apply_contact(params: dict[str, Any]) -> dict[str, Any]:
    """Add the configured contact email as a `mailto` parameter when available."""

    contact = get_contact_email()
    if contact and "mailto" not in params:
        params["mailto"] = contact
    return params


__all__ = [
    "apply_contact",
    "create_http_client",
    "get_contact_email",
    "get_user_agent",
]
