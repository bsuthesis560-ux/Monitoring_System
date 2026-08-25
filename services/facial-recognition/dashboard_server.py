"""
BSU Personnel Monitoring — Dashboard Server
============================================
Lightweight Flask web server that:
  · Serves the admin dashboard at http://localhost:5000
  · Starts / stops the facial recognition service as a managed subprocess
  · Streams service stdout to the browser via SSE
  · Reads and writes camera configuration in .env
  · Provides a REST endpoint for camera connection testing

Run directly:
  python dashboard_server.py

Or launched automatically by BSU_FaceRec_Launcher.exe (control_panel.py).

Requirements (in the same Python env as the recognition service):
  pip install flask python-dotenv opencv-python
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote as urlquote

try:
    from flask import Flask, Response, jsonify, render_template, request, stream_with_context
except ImportError:
    print("ERROR: Flask not installed.  Run: pip install flask")
    sys.exit(1)

try:
    from dotenv import dotenv_values
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

# ── paths ─────────────────────────────────────────────────────────────────────
_DIR           = Path(__file__).parent.resolve()
ENV_FILE       = _DIR / ".env"
SERVICE_SCRIPT = _DIR / "facial_recognition_service.py"
STREAM_PORT    = 5001   # port the recognition service uses for MJPEG

# ── Windows service (NSSM) management ───────────────────────────────────────────
# facial_recognition_service.py now runs as its own independent NSSM-managed
# Windows service (auto-starts at boot). The dashboard no longer spawns it as a
# child process — it controls it via the Windows Service Control Manager and
# tails the log file that NSSM redirects the service's stdout/stderr to.
FACEREC_SERVICE_NAME = "BSU-FaceRecognition"
SERVICE_LOG_FILE      = _DIR / "logs" / "service.log"
LOG_POLL_INTERVAL     = 1.0   # seconds between log-file polls

# ── shared state ──────────────────────────────────────────────────────────────
_service_logs: list = []
_log_lock      = threading.Lock()
_MAX_LOGS      = 500

_log_subscribers: list = []   # list[queue.Queue]
_sub_lock = threading.Lock()

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=str(_DIR / "templates"))
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_sc(*args: str) -> subprocess.CompletedProcess:
    """Run an `sc.exe` command against the Windows Service Control Manager."""
    return subprocess.run(
        ["sc", *args],
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def _facerec_win_state() -> str:
    """
    Query the live Windows SCM state of the BSU-FaceRecognition service.
    Returns one of: RUNNING, STOPPED, START_PENDING, STOP_PENDING,
    PAUSED, NOT_INSTALLED, UNKNOWN.
    """
    result = _run_sc("query", FACEREC_SERVICE_NAME)
    out = result.stdout or ""
    if result.returncode != 0 or "FAILED" in out.upper():
        if "1060" in out or "does not exist" in out.lower():
            return "NOT_INSTALLED"
        return "UNKNOWN"
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("STATE"):
            # e.g. "STATE              : 4  RUNNING"
            parts = line.split(":", 1)[1].strip().split(None, 1)
            if len(parts) == 2:
                return parts[1].strip()
    return "UNKNOWN"


def _load_env() -> dict:
    if _HAS_DOTENV and ENV_FILE.exists():
        return dict(dotenv_values(ENV_FILE))
    env: dict = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _save_env(data: dict):
    lines = [f"{k}={v}\n" for k, v in data.items()]
    ENV_FILE.write_text("".join(lines), encoding="utf-8")


def _append_log(line: str):
    with _log_lock:
        _service_logs.append(line)
        if len(_service_logs) > _MAX_LOGS:
            _service_logs.pop(0)
    with _sub_lock:
        for q in list(_log_subscribers):
            try:
                q.put_nowait(line)
            except Exception:
                pass


def _tail_log_file(path: Path, poll_interval: float = LOG_POLL_INTERVAL):
    """
    Continuously tail the NSSM-managed log file for the facial recognition
    service and push new lines to the dashboard's log buffer / SSE subscribers.
    Runs for the lifetime of the dashboard process, independent of whether the
    dashboard itself started the recognition service (it may already be
    running from boot via NSSM).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    last_size = 0
    while True:
        try:
            if path.exists():
                size = path.stat().st_size
                if size < last_size:
                    # File was truncated or rotated by NSSM — start over.
                    last_size = 0
                if size > last_size:
                    with path.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(last_size)
                        new_data = f.read()
                        last_size = f.tell()
                    for line in new_data.splitlines():
                        if line.strip():
                            _append_log(line)
        except Exception:
            pass
        time.sleep(poll_interval)


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return render_template("index.html", stream_port=STREAM_PORT)


@app.route("/api/service/start", methods=["POST"])
def start_service():
    state = _facerec_win_state()
    if state == "NOT_INSTALLED":
        return jsonify({"ok": False, "message": (
            f"Windows service '{FACEREC_SERVICE_NAME}' is not installed. "
            "Install it with NSSM first."
        )})
    if state in ("RUNNING", "START_PENDING"):
        return jsonify({"ok": False, "message": "Service is already running."})

    result = _run_sc("start", FACEREC_SERVICE_NAME)
    if result.returncode == 0:
        _append_log("[Dashboard] Recognition service starting…")
        return jsonify({"ok": True, "message": "Service starting…"})
    msg = (result.stderr or result.stdout or "Failed to start service.").strip()
    _append_log(f"[Dashboard] Failed to start service: {msg}")
    return jsonify({"ok": False, "message": msg})


@app.route("/api/service/stop", methods=["POST"])
def stop_service():
    state = _facerec_win_state()
    if state not in ("RUNNING", "START_PENDING"):
        return jsonify({"ok": False, "message": "Service was not running."})

    result = _run_sc("stop", FACEREC_SERVICE_NAME)
    if result.returncode == 0:
        _append_log("[Dashboard] Recognition service stopping…")
        return jsonify({"ok": True, "message": "Service stopping…"})
    msg = (result.stderr or result.stdout or "Failed to stop service.").strip()
    _append_log(f"[Dashboard] Failed to stop service: {msg}")
    return jsonify({"ok": False, "message": msg})


@app.route("/api/service/status")
def service_status():
    state = _facerec_win_state()
    return jsonify({"running": state == "RUNNING", "state": state})


@app.route("/api/service/logs")
def get_logs():
    with _log_lock:
        return jsonify({"logs": list(_service_logs)})


@app.route("/api/service/logs/stream")
def stream_logs():
    q: queue.Queue = queue.Queue(maxsize=200)
    with _sub_lock:
        _log_subscribers.append(q)

    def generate():
        with _log_lock:
            for line in _service_logs[-100:]:
                yield f"data: {json.dumps(line)}\n\n"
        try:
            while True:
                try:
                    line = q.get(timeout=25)
                    yield f"data: {json.dumps(line)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with _sub_lock:
                try:
                    _log_subscribers.remove(q)
                except ValueError:
                    pass

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/cameras", methods=["GET"])
def get_cameras():
    env = _load_env()
    cameras = []
    for i in range(1, 9):
        ip = env.get(f"CAM{i}_IP", "").strip()
        if not ip:
            break
        name = env.get(f"CAM{i}_NAME", f"Camera {i}").strip()
        cameras.append({
            "index":       i,
            "name":        name,
            "ip":          ip,
            "user":        env.get(f"CAM{i}_USER",        "admin").strip(),
            "port":        env.get(f"CAM{i}_PORT",        "554").strip(),
            "stream_path": env.get(f"CAM{i}_STREAM_PATH", "/Streaming/Channels/101").strip(),
            "stream_url":  f"http://localhost:{STREAM_PORT}/stream/{urlquote(name)}",
        })
    return jsonify({
        "cameras":    cameras,
        "api_url":    env.get("API_URL",    ""),
        "api_key":    env.get("API_KEY",    ""),
        "yolo_imgsz": env.get("YOLO_IMGSZ", "640"),
    })


@app.route("/api/cameras", methods=["POST"])
def save_cameras():
    data = request.get_json(force=True)
    env  = _load_env()

    # Clear previous per-camera keys
    for i in range(1, 9):
        for key in [f"CAM{i}_NAME", f"CAM{i}_IP", f"CAM{i}_USER",
                    f"CAM{i}_PASS", f"CAM{i}_PORT", f"CAM{i}_STREAM_PATH"]:
            env.pop(key, None)

    for cam in data.get("cameras", []):
        i = int(cam.get("index", 0))
        if not i:
            continue
        for field, env_key in [
            ("name",        f"CAM{i}_NAME"),
            ("ip",          f"CAM{i}_IP"),
            ("user",        f"CAM{i}_USER"),
            ("pass",        f"CAM{i}_PASS"),
            ("port",        f"CAM{i}_PORT"),
            ("stream_path", f"CAM{i}_STREAM_PATH"),
        ]:
            val = str(cam.get(field, "")).strip()
            if val:
                env[env_key] = val

    if "api_url"    in data: env["API_URL"]    = str(data["api_url"]).strip()
    if "api_key"    in data: env["API_KEY"]    = str(data["api_key"]).strip()
    if "yolo_imgsz" in data: env["YOLO_IMGSZ"] = str(data["yolo_imgsz"]).strip() or "640"

    _save_env(env)
    return jsonify({"ok": True})


@app.route("/api/cameras/test", methods=["POST"])
def test_camera():
    data = request.get_json(force=True)
    url  = str(data.get("url", "")).strip()
    if not url:
        return jsonify({"ok": False, "message": "No URL provided."})
    try:
        import cv2
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        deadline = time.time() + 5.0
        ok = False
        while time.time() < deadline:
            ret, _ = cap.read()
            if ret:
                ok = True
                break
        cap.release()
        msg = "Connection successful." if ok else "Stream opened but no frames received within 5 s."
        return jsonify({"ok": ok, "message": msg})
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)})


# ── background log tailer ────────────────────────────────────────────────────
# Started at import time (not just __main__) so it also runs correctly when
# the dashboard itself is launched by NSSM.
threading.Thread(target=_tail_log_file, args=(SERVICE_LOG_FILE,), daemon=True).start()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BSU FaceRec Dashboard Server")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    print(f"[Dashboard] Starting BSU FaceRec Dashboard at http://localhost:{args.port}")
    print(f"[Dashboard] Service script : {SERVICE_SCRIPT}")
    print(f"[Dashboard] Stream port    : {STREAM_PORT}")
    print("[Dashboard] Press Ctrl+C to stop.\n")

    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
