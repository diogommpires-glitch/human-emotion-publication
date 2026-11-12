"""Serve the local scientific workbench on a stable browser origin."""

from __future__ import annotations

import http.server
import json
import logging
import os
import pathlib
import socket
import stat
import tempfile
import threading
import time
import urllib.parse
import webbrowser
import zipfile
from collections.abc import Iterable
from typing import ClassVar, cast

from brain_model import model

APP_DIR = pathlib.Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"
HOST = "127.0.0.1"
PORTS = range(8766, 8776)
SIGNATURE = b"human-emotion"
IDLE_AFTER_PAGE = float(os.environ.get("HUMAN_EMOTION_IDLE", "600"))
IDLE_NO_PAGE = float(os.environ.get("HUMAN_EMOTION_STARTUP", "180"))
OPEN_BROWSER = os.environ.get("HUMAN_EMOTION_NO_BROWSER") != "1"
JSON_LIMIT = 65_536
CHECKPOINT_LIMIT = 8 * 2**30
LOGGER = logging.getLogger(__name__)


class Activity:
    def __init__(self):
        self.started = time.monotonic()
        self.last = None
        self.running = 0
        self.lock = threading.Lock()

    def touch(self):
        with self.lock:
            self.last = time.monotonic()

    def busy(self, delta):
        with self.lock:
            self.running += delta

    def expired(self, external_busy=False):
        with self.lock:
            if self.running or external_busy:
                return False
            since = self.last if self.last is not None else self.started
            timeout = IDLE_AFTER_PAGE if self.last is not None else IDLE_NO_PAGE
            return time.monotonic() - since > timeout


activity = Activity()


class RequestTooLarge(ValueError):
    pass


class RequestError(ValueError):
    """An invalid JSON request at the HTTP boundary."""


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map: ClassVar = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".mjs": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".svg": "image/svg+xml",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def _activity(self):
        return getattr(self.server, "activity", activity)

    def _json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _local(self):
        port = cast(http.server.HTTPServer, self.server).server_port
        hosts = (f"{HOST}:{port}", f"localhost:{port}")
        if self.headers.get("Host", "") not in hosts:
            self._json({"error": "Requests must be addressed to this computer."}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in tuple(f"http://{host}" for host in hosts):
            self._json({"error": "Requests must come from this application."}, 403)
            return False
        self._activity().touch()
        return True

    def _parts(self):
        return urllib.parse.urlsplit(self.path).path.strip("/").split("/")

    def _service(self, name):
        service = getattr(self.server, name, None)
        if service is None:
            raise model.SimulationError(
                f"The {name} service requires PyTorch. Install the documented runtime."
            )
        return service

    def do_GET(self):
        if self._local():
            self._guard(self._get)

    def do_HEAD(self):
        if self._local():
            self._guard(super().do_HEAD)

    def _get(self):
        parts = self._parts()
        if parts == ["__ping"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(SIGNATURE)))
            self.end_headers()
            self.wfile.write(SIGNATURE)
        elif parts == ["api", "info"]:
            self._json(model.model_info())
        elif parts == ["api", "network"]:
            self._file(APP_DIR / "brain_model" / "network.json", "application/json")
        elif parts[:2] == ["api", "sessions"]:
            self._session_get(parts[2:])
        elif parts[:2] == ["api", "experiments"] and len(parts) <= 3:
            store = self._service("experiments")
            self._json(store.get(parts[2]) if len(parts) == 3 else store.list())
        elif parts[0] == "api":
            self._json({"error": "Unknown endpoint."}, 404)
        else:
            super().do_GET()

    def _session_get(self, parts):
        store = self._service("sessions")
        if not parts:
            self._json(store.list())
        elif len(parts) == 1:
            self._json(store.get(parts[0]))
        elif len(parts) == 2 and parts[1] in ("checkpoint", "export"):
            self._export(store.export(parts[0]))
        elif len(parts) == 3 and parts[1] == "runs":
            self._json(store.get_run(parts[0], parts[2]))
        else:
            self._json({"error": "Unknown endpoint."}, 404)

    def _file(self, path, content_type):
        with path.open("rb") as source:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers()
            while chunk := source.read(1024 * 1024):
                self.wfile.write(chunk)

    def _export(self, directory):
        directory = pathlib.Path(directory)
        try:
            with self._transfer_directory("export") as temporary:
                archive = pathlib.Path(temporary) / "human-emotion-checkpoint.zip"
                with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
                    for path in sorted(directory.rglob("*")):
                        if path.is_file() and not path.is_symlink():
                            bundle.write(path, path.relative_to(directory).as_posix())
                self._file(archive, "application/zip")
        finally:
            self._service("sessions").cleanup_export(directory)

    def _transfer_directory(self, purpose):
        root = pathlib.Path(getattr(self.server, "records", APP_DIR / ".local")) / "transfers"
        root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix=f"{purpose}-", dir=root)

    def do_POST(self):
        if not self._local():
            return
        self._activity().busy(1)
        try:
            self._guard(self._post)
        finally:
            self._activity().busy(-1)
            self._activity().touch()

    def _length(self, limit):
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Send a Content-Length rather than a chunked request.")
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("Request must contain a body.")
        if length > limit:
            raise RequestTooLarge(f"Request exceeds the {limit:,}-byte limit.")
        return length

    def _body(self):
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._json({"error": "Send the request as application/json."}, 415)
            return None
        raw = self.rfile.read(self._length(JSON_LIMIT))
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise RequestError("Expected a JSON object.")
        return value

    def _post(self):
        parts = self._parts()
        if parts == ["api", "sessions", "import"]:
            self._import_checkpoint()
            return
        payload = self._body()
        if payload is None:
            return
        if parts == ["api", "simulate"]:
            self._simulate(payload)
        elif parts == ["api", "sessions"]:
            self._numerical(lambda: self._service("sessions").create(payload), 201)
        elif len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "reset":
            self._numerical(lambda: self._service("sessions").reset(parts[2], payload))
        elif parts[:2] == ["api", "experiments"]:
            self._experiment_post(parts[2:], payload)
        else:
            self._json({"error": "Unknown endpoint."}, 404)

    def _simulate(self, payload):
        store = getattr(self.server, "sessions", None)
        self._numerical(
            lambda: store.run(payload) if store is not None else model.simulate(payload)
        )

    def _numerical(self, operation, status=200):
        gate = getattr(self.server, "request_gate", None)
        if gate is not None and not gate.acquire(blocking=False):
            self._json(
                {"error": "A simulation is already running. Try again when it finishes."}, 503
            )
            return
        try:
            self._json(operation(), status)
        finally:
            if gate is not None:
                gate.release()

    def _experiment_post(self, parts, payload):
        store = self._service("experiments")
        if not parts:
            self._json(store.submit(payload), 202)
        elif len(parts) == 2 and parts[1] in ("cancel", "resume"):
            operation = store.cancel if parts[1] == "cancel" else store.resume
            self._json(operation(parts[0]))
        else:
            self._json({"error": "Unknown endpoint."}, 404)

    def _import_checkpoint(self):
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0]
        if content_type not in ("application/zip", "application/octet-stream"):
            self._json({"error": "Send the checkpoint as application/zip."}, 415)
            return
        remaining = self._length(CHECKPOINT_LIMIT)
        with self._transfer_directory("import") as temporary:
            root = pathlib.Path(temporary)
            archive = root / "checkpoint.zip"
            with archive.open("wb") as output:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("The checkpoint upload ended before Content-Length.")
                    output.write(chunk)
                    remaining -= len(chunk)
            unpack_checkpoint(archive, root / "content")
            self._numerical(
                lambda: self._service("sessions").import_checkpoint(root / "content"), 201
            )

    def _guard(self, operation):
        try:
            operation()
        except BrokenPipeError, ConnectionResetError, ConnectionAbortedError:
            LOGGER.info("client disconnected; any published model revision remains committed")
        except RequestTooLarge as exc:
            self._json({"error": str(exc)}, 413)
        except (ValueError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
            status = {"SessionConflict": 409, "CheckpointTooLarge": 413}.get(
                type(exc).__name__, 400
            )
            self._json({"error": str(exc)}, status)
        except FileNotFoundError:
            self._json({"error": "The requested model record was not found."}, 404)
        except model.SimulationError as exc:
            self._json({"error": str(exc)}, 503)
        except OSError as exc:
            if exc.errno == 28 or type(exc).__name__ == "StorageError":
                self._json({"error": "The model record could not be saved. Check disk space."}, 507)
            else:
                LOGGER.exception("local model I/O failed")
                self._json({"error": "Local I/O failed. Details are in server.log."}, 500)
        except Exception:
            LOGGER.exception("local request failed")
            self._json(
                {"error": "The request failed because of a bug. Details are in server.log."}, 500
            )

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; "
            "base-uri 'self'; frame-ancestors 'none'",
        )
        super().end_headers()

    def log_message(self, _fmt, *_args):
        pass


def unpack_checkpoint(archive: pathlib.Path, destination: pathlib.Path):
    """Check every member before extracting a bounded, data-only checkpoint."""
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > 10_000 or sum(item.file_size for item in members) > CHECKPOINT_LIMIT:
            raise RequestTooLarge("Checkpoint contents exceed the import budget.")
        names = set()
        for item in members:
            path = pathlib.PurePosixPath(item.filename)
            if (
                item.is_dir()
                or stat.S_ISLNK(item.external_attr >> 16)
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in item.filename
                or ":" in item.filename
                or path.suffix not in (".json", ".safetensors")
                or item.filename.casefold() in names
            ):
                raise ValueError("Checkpoint contains an invalid or repeated member.")
            names.add(item.filename.casefold())
        destination.mkdir()
        for item in members:
            target = destination.joinpath(*pathlib.PurePosixPath(item.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(item) as source, target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)


def probe(port):
    try:
        connection = socket.create_connection((HOST, port), timeout=0.3)
    except OSError:
        return "free"
    with connection:
        try:
            connection.settimeout(1.5)
            connection.sendall(f"GET /__ping HTTP/1.0\r\nHost: {HOST}:{port}\r\n\r\n".encode())
            data = b""
            while len(data) < 4096:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                data += chunk
            headers, _, body = data.partition(b"\r\n\r\n")
            return "ours" if b" 200 " in headers.split(b"\r\n")[0] and body == SIGNATURE else "busy"
        except OSError:
            return "busy"


def choose_port(state_path: pathlib.Path, ports: Iterable[int] = PORTS) -> int:
    if state_path.exists():
        value = json.loads(state_path.read_text(encoding="utf-8"))
        port = value.get("port") if isinstance(value, dict) else None
        if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
            raise ValueError("Saved application port must be an integer between 1024 and 65535.")
        if probe(port) == "busy":
            raise OSError(f"Saved port {port} is occupied. Close that service before restarting.")
        return port
    for port in ports:
        if probe(port) != "busy":
            return port
    raise OSError("No free local application port was found.")


class LocalServer(http.server.ThreadingHTTPServer):
    def __init__(self, address, records: pathlib.Path):
        from research import ExperimentStore

        super().__init__(address, Handler)
        self.daemon_threads = True
        self.records = records
        self.activity = Activity()
        self.request_gate = threading.BoundedSemaphore(1)
        self.sessions = None
        if model.simulator is not None:
            from brain_model.sessions import SessionStore

            self.sessions = SessionStore(records / "sessions")
        self.experiments = ExperimentStore(records / "experiments")
        self.experiments.recover()


def run_experiment_worker(store, stop, interval=0.5):
    while not stop.wait(interval):
        try:
            store.run_pending()
        except Exception:
            # Job failures are recorded by the store. Keep polling after service I/O failures.
            LOGGER.exception("experiment worker polling failed; retrying on the next poll")


def main():
    logging.basicConfig(filename=APP_DIR / "server.log", level=logging.INFO)
    state_path = APP_DIR / ".local" / "server.json"
    port = choose_port(state_path)
    if probe(port) == "ours":
        if OPEN_BROWSER:
            webbrowser.open(f"http://{HOST}:{port}/")
        return
    httpd = LocalServer((HOST, port), state_path.parent)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    pending = state_path.with_suffix(".tmp")
    pending.write_text(json.dumps({"port": port}), encoding="utf-8")
    os.replace(pending, state_path)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    stop = threading.Event()

    threading.Thread(
        target=run_experiment_worker, args=(httpd.experiments, stop), daemon=True
    ).start()
    LOGGER.info("serving the workbench at http://%s:%d", HOST, port)
    if OPEN_BROWSER:
        webbrowser.open(f"http://{HOST}:{port}/")
    try:
        while not stop.wait(1):
            if httpd.activity.expired(httpd.experiments.active_count > 0):
                break
    finally:
        stop.set()
        httpd.shutdown()
        httpd.server_close()
    LOGGER.info("idle timeout; server stopped")


if __name__ == "__main__":
    main()
