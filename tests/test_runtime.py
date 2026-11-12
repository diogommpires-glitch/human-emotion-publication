"""Exercise the real loopback HTTP boundary and stable-port policy."""

import http.client
import json
import stat
import threading
import zipfile

import pytest

import server


@pytest.fixture
def address():
    httpd = server.http.server.ThreadingHTTPServer((server.HOST, 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def request(address, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection(*address, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    status, payload, fields = response.status, response.read(), dict(response.getheaders())
    connection.close()
    return status, payload, fields


def test_local_title_and_foreign_host(address):
    status, body, headers = request(address, "GET", "/")
    assert status == 200
    assert b"<title>Human Emotion</title>" in body
    assert headers["X-Frame-Options"] == "DENY"
    status, _, _ = request(address, "GET", "/", headers={"Host": "untrusted.example"})
    assert status == 403


def test_cross_origin_post_refused_before_dispatch(address):
    status, _, _ = request(
        address,
        "POST",
        "/api/unknown",
        b"{}",
        {
            "Content-Type": "application/json",
            "Origin": "https://untrusted.example",
        },
    )
    assert status == 403


def test_head_checks_host_and_origin(address):
    status, _, _ = request(address, "HEAD", "/", headers={"Host": "foreign.example"})
    assert status == 403
    status, _, _ = request(address, "HEAD", "/", headers={"Origin": "https://foreign.example"})
    assert status == 403


def test_server_restart_recovers_interrupted_job(tmp_path):
    from research import ExperimentStore

    store = ExperimentStore(tmp_path / "experiments")
    job = store.submit({"protocol": "synthetic"})
    job["status"] = "running"
    store._save(job)
    httpd = server.LocalServer((server.HOST, 0), tmp_path)
    try:
        assert httpd.experiments.get(job["id"])["status"] == "interrupted"
        assert httpd.experiments.resume(job["id"])["status"] == "queued"
    finally:
        httpd.server_close()


def test_worker_keeps_polling_after_unexpected_failure(caplog):
    stop = threading.Event()

    class Store:
        calls = 0

        def run_pending(self):
            self.calls += 1
            if self.calls == 1:
                raise ArithmeticError("worker boundary regression")
            stop.set()

    store = Store()
    server.run_experiment_worker(store, stop, interval=0.001)
    assert store.calls == 2
    assert "worker boundary regression" in caplog.text


def test_invalid_request_shape_is_client_error(address):
    status, body, _ = request(
        address,
        "POST",
        "/api/simulate",
        b"[]",
        {
            "Content-Type": "application/json",
        },
    )
    assert status == 400
    assert "object" in json.loads(body)["error"]


def test_oversized_json_refused_before_read(address):
    status, _, _ = request(
        address,
        "POST",
        "/api/simulate",
        b"{}",
        {
            "Content-Type": "application/json",
            "Content-Length": "65537",
        },
    )
    assert status == 413


def test_saved_port_never_silently_changes(tmp_path, address):
    state = tmp_path / "server.json"
    state.write_text(json.dumps({"port": address[1]}))
    # The existing fixture is this app and may be reused at the same origin.
    assert server.choose_port(state, [address[1] + 1]) == address[1]


def test_invalid_port_state_is_not_a_fresh_install(tmp_path):
    state = tmp_path / "server.json"
    state.write_text('{"port": false}')
    with pytest.raises(ValueError, match="port"):
        server.choose_port(state, [8766])


def test_checkpoint_symlink_rejected_before_extraction(tmp_path):
    archive = tmp_path / "checkpoint.zip"
    member = zipfile.ZipInfo("manifest.json")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, "../../other.json")
    destination = tmp_path / "unpacked"
    with pytest.raises(ValueError, match="invalid"):
        server.unpack_checkpoint(archive, destination)
    assert not destination.exists()
