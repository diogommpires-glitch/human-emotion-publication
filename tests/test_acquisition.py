"""Public download redirects must retain the same allowlist as the source URL."""

import hashlib
import urllib.request

import pytest

from research.datasets import DatasetFile, PublicRedirect, acquire_file


def test_acquisition_installs_redirect_guard_before_opening(tmp_path, monkeypatch):
    payload = b"synthetic public metadata"
    item = DatasetFile(
        "metadata.json",
        "https://raw.githubusercontent.com/example/metadata.json",
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        "CC0",
    )

    def build_opener(*handlers):
        assert len(handlers) == 1 and isinstance(handlers[0], PublicRedirect)

        class Opener:
            def open(self, url, **kwargs):
                return handlers[0].redirect_request(
                    urllib.request.Request(url),
                    None,
                    302,
                    "Found",
                    {},
                    "https://127.0.0.1/private",
                )

        return Opener()

    def unguarded_open(*args, **kwargs):
        pytest.fail("acquisition bypassed the validated redirect opener")

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    monkeypatch.setattr(urllib.request, "urlopen", unguarded_open)
    with pytest.raises(ValueError, match="approved public"):
        acquire_file(item, tmp_path / item.path, lambda: None)
    assert list(tmp_path.iterdir()) == []
