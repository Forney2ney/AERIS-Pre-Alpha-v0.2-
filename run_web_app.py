"""
run_web_app.py
================
Starts the AERIS local web interface and opens it in your browser
automatically. Run this instead of start_here.py if you'd rather use a
visual interface than the text menu — both talk to exactly the same
underlying engine.

How to run: python3 run_web_app.py
Then just leave this terminal window open and use your browser. Press
Ctrl+C in the terminal to stop the server when you're done.
"""

import sys
import os
import threading
import webbrowser
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aeris.webapp.server import run

PORT = 8765


def _open_browser():
    time.sleep(1.0)  # give the server a moment to start listening
    webbrowser.open(f"http://127.0.0.1:{PORT}")


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()
    run(port=PORT)
