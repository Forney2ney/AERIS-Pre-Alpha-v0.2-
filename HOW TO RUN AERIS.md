# How to Run AERIS — Complete Beginner Instructions

This assumes you have never used a terminal or installed Python before.
Follow the section for your computer (Windows or Mac). It takes about
10–15 minutes the first time; after that, running it takes 10 seconds.

---

## STEP 1 — Install Python (one-time only)

### Windows
1. Go to https://www.python.org/downloads/ in your browser.
2. Click the big yellow "Download Python 3.x.x" button.
3. Open the downloaded file.
4. **Important:** on the very first install screen, tick the box that says
   **"Add python.exe to PATH"** at the bottom, THEN click "Install Now."
5. When it finishes, click "Close."

### Mac
1. Go to https://www.python.org/downloads/ in your browser.
2. Click "Download Python 3.x.x for macOS."
3. Open the downloaded `.pkg` file and click through the installer
   (Continue → Continue → Agree → Install), entering your Mac password
   when asked.

**Check it worked:** open a terminal (see Step 2) and type:
```
python3 --version
```
You should see something like `Python 3.12.x`. If you see an error instead,
restart your computer and try again — this fixes it almost every time.

---

## STEP 2 — Open a terminal in the AERIS folder

1. Unzip the AERIS project folder somewhere easy to find, like your
   Desktop (right-click the zip → "Extract All" on Windows, or just
   double-click it on Mac).
2. You should now have a folder called `AERIS_project` containing
   `start_here.py`, `requirements.txt`, and an `aeris` folder inside it.

### Windows
1. Open the `AERIS_project` folder in File Explorer.
2. Click once in the address bar at the top (where the folder path is
   shown), type `cmd`, and press Enter. A black terminal window opens,
   already pointed at the right folder.

### Mac
1. Open the `AERIS_project` folder in Finder.
2. Right-click (or two-finger click) inside the folder and choose
   **"New Terminal at Folder"**. (If you don't see that option: open the
   Terminal app from Launchpad, type `cd ` with a trailing space, then drag
   the `AERIS_project` folder from Finder into the terminal window, and
   press Enter.)

You'll know you're in the right place if typing `dir` (Windows) or `ls`
(Mac) shows `start_here.py` in the list.

---

## STEP 3 — Install the two required packages (one-time only)

In that same terminal window, type exactly this and press Enter:

```
pip install -r requirements.txt
```

(On Mac, if that gives an error, try `pip3 install -r requirements.txt`
instead.)

You'll see some text scroll by installing `numpy` and `scipy` — this is
normal and only needs to happen once. Wait for it to finish and return you
to a blank prompt line.

---

## STEP 4 — Run AERIS

You have two ways to run it. Both use the exact same engine underneath —
pick whichever you prefer.

### Option A: the web interface (recommended)

Type:
```
python3 run_web_app.py
```
(On Windows, if that gives an error, try `python run_web_app.py` instead.)

A browser tab should open automatically at `http://127.0.0.1:8765`. If it
doesn't open by itself, open your browser and type that address in
manually. You'll see a sidebar with four sections (Airfoil, Winglet,
Aircraft, Logbook) — click one, fill in the numbers (or leave the defaults),
and click "Run optimization." A live chart shows the optimizer improving
its design in real time. When it finishes, you'll see the best design's
numbers, a "run ID," and — for the airfoil and aircraft modules — buttons
to download a DXF or STL file for CAD/3D printing.

**Leave the terminal window open** while you use the browser — closing it
stops the server. When you're done, go back to the terminal and press
`Ctrl+C` (or just close the window).

### Option B: the plain text menu

Type:
```
python3 start_here.py
```
(On Windows, if that gives an error, try `python start_here.py` instead.)

You'll see a menu like this:

```
============================================================
  AERIS — Aerospace Engineering Reasoning & Intelligent Synthesis
============================================================
  1) Design + optimize an AIRFOIL (with biomimicry)
  2) Design + optimize a WINGLET
  3) Design + optimize a FULL AIRCRAFT (flying wing)
  4) View your logbook (past runs)
  5) Record a real test result against a past run
  6) Exit

Type a number (1-6) and press Enter:
```

Type a number and press Enter. Options 1 and 2 take about 10–30 seconds.
Option 3 takes 1–3 minutes (it's searching over 10 variables at once — this
is normal, just let it run). When it finishes, it prints the best design it
found and a "run ID" (a short code like `8fe8bccc`).

**Write that run ID down** (or take a screenshot) — you'll need it later
when you test the design physically and want to record real measurements
(menu option 5).

To leave the program at any time, choose option 6, or close the terminal
window.

---

## STEP 5 — Running it again later

You do NOT need to repeat Steps 1–3 again. Just:
1. Open a terminal in the `AERIS_project` folder (Step 2).
2. Type `python3 run_web_app.py` (or `python3 start_here.py` for the text menu).

---

## Common problems

- **"python3 is not recognized" / "command not found"** — Python wasn't
  added to PATH during install. Reinstall Python (Step 1) and make sure to
  tick "Add python.exe to PATH" on Windows, or just try `python` instead of
  `python3`.
- **"pip is not recognized"** — try `python3 -m pip install -r requirements.txt`
  instead.
- **A wall of red text mentioning "numpy" or "scipy"** — Step 3 didn't
  finish successfully. Check your internet connection and re-run
  `pip install -r requirements.txt`.
- **The window flashes and closes immediately** — you double-clicked
  `start_here.py` instead of running it from a terminal. Follow Step 2 and
  Step 4 exactly; don't double-click the file.
- **It says "No module named aeris"** — you're running the command from
  the wrong folder. Make sure your terminal is open *inside*
  `AERIS_project` (the folder that directly contains `start_here.py`), not
  one level above or below it.
- **The browser says "can't connect" / "site can't be reached"** — the
  server likely hasn't finished starting yet, or the terminal window was
  closed. Make sure the terminal still shows "AERIS web UI running at
  http://127.0.0.1:8765", then reload the browser tab.
- **"Address already in use" when running `run_web_app.py`** — you already
  have it running in another terminal window. Close that window first, or
  just reuse the browser tab that's already open.
