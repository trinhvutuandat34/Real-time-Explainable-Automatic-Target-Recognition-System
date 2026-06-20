"""
Module E — iPhone Camera Streamer

Serves a mobile-optimised web page to an iPhone/iPad browser.
The page captures the rear camera via getUserMedia and streams JPEG frames
to this server over WebSocket.  The Streamlit dashboard polls /frame at
~15 FPS to get the latest image.

Quick start:
    python modules/module_e_streamer.py --ngrok   # HTTPS via free ngrok tunnel
    python modules/module_e_streamer.py           # HTTP on LAN (same WiFi)

getUserMedia on iOS Safari requires HTTPS unless the page is served from
localhost.  --ngrok creates a free HTTPS tunnel so the iPhone can open
the capture page without any certificate setup.

Dashboard reads frames from:
    http://localhost:7860/frame   (GET → JPEG bytes, 204 if none yet)
    http://localhost:7860/status  (GET → JSON stats)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import socket
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response

# ---------------------------------------------------------------------------
# iPhone capture page (served at GET /)
# ---------------------------------------------------------------------------

_CAPTURE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,
      maximum-scale=1,user-scalable=no,viewport-fit=cover">
<title>REATS Camera</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d0d0d; color: #f0f0f0;
         font-family: -apple-system, system-ui, sans-serif;
         overflow: hidden; height: 100svh; }
  #video { position: fixed; inset: 0; width: 100%; height: 100%;
           object-fit: cover; }
  #hud   { position: fixed; top: 0; left: 0; right: 0;
           padding: env(safe-area-inset-top, 12px) 16px 12px;
           background: linear-gradient(180deg,rgba(0,0,0,.75) 0%,transparent);
           display: flex; flex-direction: column; gap: 4px; }
  .row   { display: flex; align-items: center; gap: 8px; font-size: 13px; }
  .dot   { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0;
           background: #e55; transition: background .3s; }
  .dot.ok{ background: #4d4; animation: blink 1.4s ease-in-out infinite; }
  @keyframes blink { 50% { opacity: .35; } }
  #hint  { position: fixed; bottom: calc(env(safe-area-inset-bottom,0px)+18px);
           left: 0; right: 0; text-align: center;
           font-size: 12px; color: rgba(255,255,255,.45); letter-spacing: .5px; }
  canvas { display: none; }
</style>
</head>
<body>
<video id="video" autoplay playsinline muted></video>
<canvas id="canvas"></canvas>
<div id="hud">
  <div class="row"><div class="dot" id="dot"></div><span id="status">Starting camera…</span></div>
  <div class="row" id="fps-row" style="display:none">
    <span id="fps">0 fps</span>
    <span style="opacity:.5">→ REATS dashboard</span>
  </div>
</div>
<div id="hint">Point rear camera at target &nbsp;·&nbsp; REATS live feed</div>
<script>
'use strict';
const TARGET_FPS = 15;
const JPEG_Q    = 0.78;     // 0–1; lower = smaller frames, faster

const video  = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');
const dot    = document.getElementById('dot');
const status = document.getElementById('status');
const fpsRow = document.getElementById('fps-row');
const fpsEl  = document.getElementById('fps');

let ws, sending = false, fCount = 0, fTs = performance.now();

/* ── Camera ── */
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: 'environment' },
        width:  { ideal: 1280 },
        height: { ideal: 720 },
      }
    });
    video.srcObject = stream;
    await video.play();
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    status.textContent = 'Camera ready — connecting…';
    connect();
  } catch (e) {
    status.textContent = 'Camera error: ' + e.message;
    console.error(e);
  }
}

/* ── WebSocket (auto-reconnect) ── */
function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws');

  ws.onopen = () => {
    dot.className = 'dot ok';
    status.textContent = 'Streaming to dashboard';
    fpsRow.style.display = 'flex';
    if (!sending) { sending = true; schedule(); }
  };
  ws.onclose = () => {
    dot.className = 'dot';
    status.textContent = 'Reconnecting…';
    sending = false;
    setTimeout(connect, 2500);
  };
  ws.onerror = () => ws.close();
}

/* ── Send loop ── */
function schedule() {
  setTimeout(sendFrame, Math.round(1000 / TARGET_FPS));
}

function sendFrame() {
  if (!sending) return;
  if (ws.readyState === WebSocket.OPEN && video.readyState >= 2) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ws.send(canvas.toDataURL('image/jpeg', JPEG_Q));
    fCount++;
    const now = performance.now();
    if (now - fTs >= 1000) {
      fpsEl.textContent = fCount + ' fps';
      fCount = 0; fTs = now;
    }
  }
  schedule();
}

startCamera();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="REATS iPhone Streamer", docs_url=None, redoc_url=None)

# Shared state — writes are atomic in CPython (GIL), reads are safe
_latest_jpeg: bytes | None = None
_frame_count: int  = 0
_frame_ts:    float = 0.0
_ws_clients:  int  = 0


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(content=_CAPTURE_HTML)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    global _latest_jpeg, _frame_count, _frame_ts, _ws_clients
    await ws.accept()
    _ws_clients += 1
    logging.info("iPhone connected  (total clients: %d)", _ws_clients)
    try:
        while True:
            raw = await ws.receive_text()          # "data:image/jpeg;base64,<b64>"
            if "," in raw:
                raw = raw.split(",", 1)[1]
            try:
                _latest_jpeg = base64.b64decode(raw)
                _frame_count += 1
                _frame_ts    = time.monotonic()
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients -= 1
        logging.info("iPhone disconnected  (total clients: %d)", _ws_clients)


@app.get("/frame")
async def get_frame() -> Response:
    """Return latest JPEG frame (HTTP 204 if none received yet)."""
    if _latest_jpeg is None:
        return Response(status_code=204)
    return Response(
        content=_latest_jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache"},
    )


@app.get("/status")
async def get_status() -> JSONResponse:
    age = round(time.monotonic() - _frame_ts, 2) if _frame_ts else None
    return JSONResponse({
        "clients":     _ws_clients,
        "frame_count": _frame_count,
        "frame_age_s": age,
        "streaming":   _ws_clients > 0 and age is not None and age < 2.0,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="REATS iPhone Camera Streamer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # HTTPS tunnel — required for iOS getUserMedia on non-localhost
  python modules/module_e_streamer.py --ngrok

  # LAN only (works if iPhone and laptop on same WiFi AND iOS trusts the cert)
  python modules/module_e_streamer.py

  # Custom port
  python modules/module_e_streamer.py --port 8080 --ngrok
""",
    )
    parser.add_argument("--host",        default="0.0.0.0",
                        help="bind host (default: 0.0.0.0)")
    parser.add_argument("--port",        type=int, default=7860,
                        help="HTTP port (default: 7860)")
    parser.add_argument("--ngrok",       action="store_true",
                        help="open ngrok HTTPS tunnel (recommended for iOS)")
    parser.add_argument("--ngrok-token", default="", metavar="TOKEN",
                        help="ngrok authtoken (or set NGROK_AUTHTOKEN env var)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [streamer] %(message)s",
                        datefmt="%H:%M:%S")

    SEP = "=" * 58
    if args.ngrok:
        try:
            from pyngrok import ngrok, conf as ngrok_conf   # type: ignore
            if args.ngrok_token:
                ngrok_conf.get_default().auth_token = args.ngrok_token
            tunnel    = ngrok.connect(args.port, "http")
            phone_url = tunnel.public_url
        except ImportError:
            print("pyngrok not installed — run:  pip install pyngrok")
            return
        except Exception as e:
            print(f"ngrok error: {e}")
            print("Falling back to LAN-only mode.")
            phone_url = f"http://{_local_ip()}:{args.port}"
    else:
        phone_url = f"http://{_local_ip()}:{args.port}"

    print(f"\n{SEP}")
    print(f"  iPhone URL  →  {phone_url}")
    print(f"  (open this URL in Safari on your iPhone)")
    if not args.ngrok:
        print(f"  NOTE: iOS requires HTTPS for camera access.")
        print(f"  If camera doesn't start, re-run with --ngrok")
    print()
    print(f"  Dashboard reads frames from:")
    print(f"  http://localhost:{args.port}/frame")
    print(f"{SEP}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
