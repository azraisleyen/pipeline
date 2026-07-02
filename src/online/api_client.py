from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from src.common.exceptions import PipelineError
from src.online.request_models import OnlineFrameRequest


class OnlineApiError(PipelineError):
    pass


class OnlineApiRetryableError(OnlineApiError):
    pass


@dataclass
class RetryPolicy:
    attempts: int = 3
    backoff_seconds: float = 0.5
    max_backoff_seconds: float = 5.0


class OnlineApiClient:
    """Generic HTTP client for TEKNOFEST-style fetch/submit simulation APIs.

    Endpoint names are config-driven because final competition URLs may differ.
    The default convention is:
      GET  {base_url}/frame
      POST {base_url}/packet
    """

    def __init__(
        self,
        base_url: str = "",
        token: str = "",
        *,
        token_env: str = "TEKNOFEST_API_TOKEN",
        timeout_seconds: float = 10.0,
        fetch_endpoint: str = "/frame",
        submit_endpoint: str = "/packet",
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/" if base_url else ""
        self.token = token or os.getenv(token_env, "")
        self.token_env = token_env
        self.timeout_seconds = float(timeout_seconds)
        self.fetch_endpoint = fetch_endpoint
        self.submit_endpoint = submit_endpoint
        self.retry_policy = retry_policy or RetryPolicy()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OnlineApiClient":
        online = config.get("online", config)
        retry = online.get("retry", {})
        return cls(
            base_url=online.get("base_url", ""),
            token=online.get("token", ""),
            token_env=online.get("token_env", "TEKNOFEST_API_TOKEN"),
            timeout_seconds=online.get("timeout_seconds", 10),
            fetch_endpoint=online.get("fetch_endpoint", "/frame"),
            submit_endpoint=online.get("submit_endpoint", "/packet"),
            retry_policy=RetryPolicy(
                attempts=int(retry.get("attempts", 3)),
                backoff_seconds=float(retry.get("backoff_seconds", 0.5)),
                max_backoff_seconds=float(retry.get("max_backoff_seconds", 5.0)),
            ),
        )

    def fetch_frame(self) -> OnlineFrameRequest | None:
        payload = self._request_json("GET", self.fetch_endpoint)
        if payload is None or payload.get("end_of_stream") or payload.get("status") in {"done", "finished", "no_frame"}:
            return None
        return OnlineFrameRequest.from_payload(payload, timeout_seconds=self.timeout_seconds, token=self.token)

    def submit_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        payload = self._request_json("POST", self.submit_endpoint, packet)
        return payload or {}

    def _request_json(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not self.base_url:
            raise OnlineApiError("online.base_url is not configured")
        url = urljoin(self.base_url, endpoint.lstrip("/"))

        def send_once() -> dict[str, Any] | None:
            body = json.dumps(payload).encode("utf-8") if payload is not None else None
            headers = {"Accept": "application/json"}
            if body is not None:
                headers["Content-Type"] = "application/json"
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            request = Request(url, data=body, headers=headers, method=method)
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read()
                    if not raw:
                        return {}
                    return json.loads(raw.decode("utf-8"))
            except HTTPError as exc:
                message = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
                if 500 <= exc.code < 600 or exc.code in {408, 429}:
                    raise OnlineApiRetryableError(f"HTTP {exc.code} from {url}: {message}") from exc
                raise OnlineApiError(f"HTTP {exc.code} from {url}: {message}") from exc
            except URLError as exc:
                raise OnlineApiRetryableError(f"Network error calling {url}: {exc.reason}") from exc
            except TimeoutError as exc:
                raise OnlineApiRetryableError(f"Timeout calling {url}") from exc
            except json.JSONDecodeError as exc:
                raise OnlineApiError(f"Invalid JSON response from {url}: {exc}") from exc

        delay = self.retry_policy.backoff_seconds
        last_exc: Exception | None = None
        for attempt in range(1, max(1, self.retry_policy.attempts) + 1):
            try:
                return send_once()
            except OnlineApiRetryableError as exc:
                last_exc = exc
                if attempt >= self.retry_policy.attempts:
                    break
                time.sleep(delay)
                delay = min(delay * 2.0, self.retry_policy.max_backoff_seconds)
        assert last_exc is not None
        raise last_exc
