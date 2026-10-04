"""Launch the web app (browser, reader, job queue).

    python scripts/serve.py                 # http://127.0.0.1:8000, opens a browser
    python scripts/serve.py --port 9000 --no-browser
    python scripts/serve.py --host 0.0.0.0  # reachable from the LAN

Binds to 127.0.0.1 by default. The app can queue jobs that run arbitrary
scripts against the corpus and can read files under data/ and models/, so it
should not be exposed to a network without thinking about it first -- there is
no authentication.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _probe(url: str) -> str:
    """'ours' if this app answers at url, 'other' if something else does, else 'free'."""
    import json
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url + "/api/jobs?limit=1", timeout=2) as r:
            body = json.loads(r.read() or b"{}")
        return "ours" if isinstance(body, dict) and "items" in body else "other"
    except urllib.error.HTTPError:
        return "other"
    except (urllib.error.URLError, OSError, ValueError):
        return "free"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true",
                    help="do not open a browser window on start")
    ap.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    args = ap.parse_args()

    import uvicorn

    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    # Already running? Then this launch just opens it. Double-clicking the
    # shortcut a second time used to start a second server that lost the race
    # for the port but kept a job worker alive on the same queue.
    state = _probe(url)
    if state == "ours":
        print(f"Latin Library is already running at {url} -- opening it.")
        if not args.no_browser:
            webbrowser.open(url)
        return
    if state == "other":
        sys.exit(f"Port {args.port} is taken by something else; try --port {args.port + 1}.")
    # ASCII only: the Windows console this launches in is cp1252, and a stray
    # arrow in a startup banner is a silly way to fail to start.
    print(f"Latin Library -> {url}")
    if not args.no_browser and not args.reload:
        # Fire slightly late so the browser does not land on a connection error
        # while uvicorn is still binding the socket.
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run("web.server:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
