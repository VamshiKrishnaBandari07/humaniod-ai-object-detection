"""Local page for uploading two recordings and reading the change report."""

from __future__ import annotations

import json
import sys
import threading
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 250 * 1024 * 1024
FILE_NAMES = {
    "report.md",
    "report.json",
    "preview.png",
    "scene.json",
    "before_frame.png",
    "after_frame.png",
    "scene_before.ply",
    "scene_after.ply",
    "changes.ply",
    "viewer.html",
    "before.mp4",
    "after.mp4",
    "intrinsics.json",
}

_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}
_ACTIVE: str | None = None


def workspace() -> Path:
    cwd = Path.cwd()
    if (cwd / "spatial_change").is_dir():
        return cwd
    return ROOT


def serve(port: int = 8765, open_browser: bool = True) -> None:
    host = "127.0.0.1"
    httpd = ThreadingHTTPServer((host, port), App)
    url = f"http://{host}:{port}"
    print(f"Field memory is open at {url}", flush=True)
    print("Leave this window running. Comparisons stay on this computer.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


class App(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", (STATIC / "index.html").read_bytes())
                return
            if path == "/favicon.ico":
                self._send(204, "image/x-icon", b"")
                return
            if path == "/api/sample":
                self._send_json(self._sample_payload())
                return
            if path == "/api/active":
                with _LOCK:
                    active = _ACTIVE
                    job = None if active is None else _JOBS.get(active)
                self._send_json({"id": None} if job is None else _public_job(job))
                return
            if path.startswith("/api/jobs/"):
                self._job_get(path)
                return
            self._send_json({"error": "Not found"}, status=404)
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/demo":
                job = _start_job(kind="demo")
                self._send_json({"id": job["id"]}, status=202)
                return
            if path == "/api/compare":
                fields = _read_form(self)
                job = _start_job(kind="compare", fields=fields)
                self._send_json({"id": job["id"]}, status=202)
                return
            self._send_json({"error": "Not found"}, status=404)
        except RuntimeError as exc:
            status = 409 if "already running" in str(exc) else 400
            self._send_json({"error": str(exc)}, status=status)
        except ValueError as exc:
            message = str(exc)
            status = 413 if "large" in message else 400
            self._send_json({"error": message}, status=status)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _job_get(self, path: str) -> None:
        parts = [unquote(part) for part in path.split("/") if part]
        # api / jobs / id / ...
        if len(parts) < 3:
            self._send_json({"error": "Not found"}, status=404)
            return
        job_id = parts[2]
        directory = _job_dir(job_id)
        if len(parts) == 3:
            if job_id == "sample":
                self._send_json(self._sample_payload())
                return
            with _LOCK:
                job = _JOBS.get(job_id)
            if job is None:
                self._send_json({"error": "Unknown comparison"}, status=404)
                return
            self._send_json(_public_job(job))
            return
        target = parts[3]
        if target == "scene":
            self._send_json(_load_scene(directory))
            return
        files = {
            "preview": "preview.png",
            "before": "before_frame.png",
            "after": "after_frame.png",
        }
        if target in files:
            self._send_file(directory / files[target])
            return
        if target == "file" and len(parts) == 5 and parts[4] in FILE_NAMES:
            self._send_file(directory / parts[4])
            return
        self._send_json({"error": "Not found"}, status=404)

    def _sample_payload(self) -> dict:
        directory = _sample_dir()
        report_path = directory / "report.json"
        if not report_path.exists():
            raise FileNotFoundError("No comparison yet. Upload two recordings, or run the synthetic room.")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return {"id": "sample", "status": "done", "log": [], "error": None, "report": report}

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send_json({"error": f"Missing {path.name}"}, status=404)
            return
        kind = {
            ".png": "image/png",
            ".json": "application/json",
            ".md": "text/markdown; charset=utf-8",
            ".html": "text/html; charset=utf-8",
            ".ply": "application/octet-stream",
            ".mp4": "video/mp4",
        }.get(path.suffix.lower(), "application/octet-stream")
        self._send(200, kind, path.read_bytes())

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body)

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _start_job(kind: str, fields: dict | None = None) -> dict:
    global _ACTIVE
    with _LOCK:
        if _ACTIVE is not None and _JOBS[_ACTIVE]["status"] == "running":
            raise RuntimeError("A comparison is already running.")
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "status": "running",
            "kind": kind,
            "log": [],
            "error": None,
            "report": None,
            "dir": str(workspace() / "runs" / "ui" / job_id),
        }
        _JOBS[job_id] = job
        _ACTIVE = job_id
    try:
        if kind == "compare":
            _stash_uploads(job, fields or {})
    except Exception:
        with _LOCK:
            job["status"] = "error"
            if _ACTIVE == job_id:
                _ACTIVE = None
        raise
    thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    thread.start()
    return job


def _stash_uploads(job: dict, fields: dict) -> None:
    before = _file_field(fields, "before")
    after = _file_field(fields, "after")
    if before is None or after is None:
        raise RuntimeError("Choose both recordings: the earlier visit and the later visit.")
    out = Path(job["dir"])
    out.mkdir(parents=True, exist_ok=True)
    before_path = out / _safe_video_name(before["filename"], "before")
    after_path = out / _safe_video_name(after["filename"], "after")
    before_path.write_bytes(before["data"])
    after_path.write_bytes(after["data"])
    job["before"] = str(before_path)
    job["after"] = str(after_path)
    job["fov"] = _float_field(fields, "fov", 62.0)
    job["max_frames"] = int(_float_field(fields, "max_frames", 12))
    intrinsics = _file_field(fields, "intrinsics")
    if intrinsics and intrinsics["data"]:
        path = out / "intrinsics.json"
        path.write_bytes(intrinsics["data"])
        job["intrinsics"] = str(path)
    else:
        job["intrinsics"] = None


def _run_job(job_id: str) -> None:
    with _LOCK:
        job = _JOBS[job_id]
    try:
        _append(job, "Starting.")
        if job["kind"] == "demo":
            _run_demo(job)
        else:
            _run_compare(job)
        report_path = Path(job["dir"]) / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
        with _LOCK:
            job["report"] = report
            job["status"] = "done"
    except Exception as exc:
        _append(job, str(exc))
        with _LOCK:
            job["status"] = "error"
            job["error"] = str(exc)
            job["trace"] = traceback.format_exc()


def _run_demo(job: dict) -> None:
    from spatial_change.pipeline import run_comparison
    from spatial_change.synth import generate_dataset

    out = Path(job["dir"])
    out.mkdir(parents=True, exist_ok=True)
    _append(job, "Rendering a synthetic room with one moved, one removed, and one added object.")
    with _capture(job):
        dataset = generate_dataset(out, frames=12)
        run_comparison(
            dataset["before"],
            dataset["after"],
            str(out),
            intrinsics_path=dataset["intrinsics"],
            max_frames=12,
            ground_truth=dataset["ground_truth"],
        )


def _run_compare(job: dict) -> None:
    from spatial_change.pipeline import run_comparison

    with _capture(job):
        run_comparison(
            job["before"],
            job["after"],
            job["dir"],
            intrinsics_path=job.get("intrinsics"),
            fov_deg=float(job.get("fov", 62.0)),
            max_frames=int(job.get("max_frames", 12)),
        )


def _public_job(job: dict) -> dict:
    with _LOCK:
        return {
            "id": job["id"],
            "status": job["status"],
            "log": list(job["log"][-40:]),
            "error": job["error"],
            "report": job["report"],
        }


def _append(job: dict, line: str, echo: bool = True) -> None:
    text = line.strip()
    if not text:
        return
    with _LOCK:
        job["log"].append(text)
    if echo:
        print(text, flush=True)


class _capture:
    def __init__(self, job: dict) -> None:
        self.job = job
        self.original = sys.stdout
        self.buffer = ""

    def __enter__(self) -> "_capture":
        sys.stdout = self
        return self

    def __exit__(self, *_) -> None:
        sys.stdout = self.original
        if self.buffer.strip():
            _append(self.job, self.buffer)

    def write(self, text: str) -> int:
        self.original.write(text)
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            _append(self.job, line, echo=False)
        return len(text)

    def flush(self) -> None:
        self.original.flush()


def _sample_dir() -> Path:
    ui = workspace() / "runs" / "ui"
    reports = []
    if ui.is_dir():
        reports = [path / "report.json" for path in ui.iterdir() if (path / "report.json").is_file()]
    if reports:
        return max(reports, key=lambda path: path.stat().st_mtime).parent
    return workspace() / "runs" / "demo"


def _job_dir(job_id: str) -> Path:
    if job_id == "sample":
        return _sample_dir()
    if not job_id.isalnum():
        raise FileNotFoundError("Unknown comparison")
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is not None:
        return Path(job["dir"])
    candidate = workspace() / "runs" / "ui" / job_id
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError("Unknown comparison")


def _load_scene(directory: Path) -> dict:
    scene_path = directory / "scene.json"
    if scene_path.exists():
        return json.loads(scene_path.read_text(encoding="utf-8"))
    html_path = directory / "viewer.html"
    if not html_path.exists():
        raise FileNotFoundError("The shared map is not ready yet.")
    html = html_path.read_text(encoding="utf-8")
    marker = "const SCENE = "
    start = html.find(marker)
    if start < 0:
        raise FileNotFoundError("The shared map is missing from the viewer.")
    start += len(marker)
    end = html.find(";\nconst summary", start)
    if end < 0:
        end = html.find(";\r\nconst summary", start)
    if end < 0:
        raise FileNotFoundError("Could not read the shared map.")
    return json.loads(html[start:end])


def _read_form(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        raise ValueError("The form was empty.")
    if length > MAX_UPLOAD:
        raise ValueError("Those recordings are too large. Keep the pair under 250 MB.")
    body = bytearray()
    remaining = length
    while remaining:
        chunk = handler.rfile.read(min(remaining, 1024 * 1024))
        if not chunk:
            break
        body.extend(chunk)
        remaining -= len(chunk)
    return _parse_multipart(bytes(body), handler.headers.get("Content-Type", ""))


def _parse_multipart(body: bytes, content_type: str) -> dict:
    if "boundary=" not in content_type:
        raise ValueError("Expected a multipart upload.")
    boundary = content_type.split("boundary=", 1)[1].split(";", 1)[0].strip().strip('"')
    marker = b"--" + boundary.encode("utf-8", "replace")
    fields: dict[str, list[dict]] = {}
    for part in body.split(marker):
        if part.startswith(b"\r\n"):
            part = part[2:]
        if not part or part.startswith(b"--"):
            continue
        header_blob, separator, data = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]
        headers = header_blob.decode("utf-8", "replace")
        disposition = ""
        for line in headers.split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                disposition = line.split(":", 1)[1].strip()
        name = _disposition_value(disposition, "name")
        if not name:
            continue
        fields.setdefault(name, []).append({
            "filename": _disposition_value(disposition, "filename"),
            "data": data,
        })
    return fields


def _disposition_value(header: str, key: str) -> str | None:
    token = key + "="
    index = header.find(token)
    if index < 0:
        return None
    rest = header[index + len(token):].strip()
    if rest.startswith('"'):
        end = rest.find('"', 1)
        return rest[1:end] if end > 0 else rest[1:]
    return rest.split(";", 1)[0].strip() or None


def _file_field(fields: dict, name: str) -> dict | None:
    items = fields.get(name) or []
    for item in items:
        if item.get("filename") and item.get("data"):
            return item
    return None


def _float_field(fields: dict, name: str, default: float) -> float:
    items = fields.get(name) or []
    if not items or not items[0].get("data"):
        return default
    try:
        return float(items[0]["data"].decode("utf-8", "replace").strip())
    except ValueError:
        return default


def _safe_video_name(filename: str | None, stem: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}:
        suffix = ".mp4"
    return stem + suffix
