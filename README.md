# AERIS — Quick Start

**You do not need to know how to code to use this.** Everything you need is
in `start_here.py`. See "HOW TO RUN AERIS.md" for a full, step-by-step guide
with no assumed coding knowledge.

## The short version

1. Install Python (one-time, see the guide).
2. Open a terminal in this folder.
3. Run: `pip install -r requirements.txt` (one-time).
4. Pick ONE of these:
   - **Text menu:** run `python3 start_here.py`, then type a number.
   - **Web interface (recommended):** run `python3 run_web_app.py` — this
     opens a proper visual interface in your browser automatically, with
     live optimization progress charts, a logbook page, and CAD file
     downloads. Both talk to the exact same underlying engine.

Full details, including what to do if something goes wrong, are in
`HOW TO RUN AERIS.md`.

## What this actually is

A small computational engineering tool with three modules:

- **Airfoil + biomimicry designer** — generates and optimizes a 2D wing
  section shape, including optional whale-fin-inspired tubercles.
- **Adaptive winglet designer** — optimizes a winglet added to a wing.
- **Full aircraft (flying-wing) designer** — optimizes an entire wing shape
  at once (10 variables), built out of the physics from the first two
  modules.

Every design it produces gets saved to a logbook file
(`aeris_logbook.json`) with a unique ID. Once you physically build and test
a design (wind tunnel, thrust stand, glide test), you feed the real
measured numbers back in (through the menu or the web UI), and it shows
you exactly how far off the prediction was. **That comparison is the
actual point of the whole project** — see `aeris/ARCHITECTURE.md` for why.

Once you've optimized an airfoil or a full aircraft, you can download it as
a **DXF** (2D profile, for CAD/laser cutting) or **STL** (3D solid, ready to
3D print) file — from the web UI's export buttons, or by calling
`export_airfoil_cad()` / `export_aircraft_cad()` directly in Python (see
`aeris/ARCHITECTURE.md`).

## Where things live

```
AERIS_project/
  start_here.py             <- text-menu interface (no browser needed)
  run_web_app.py             <- opens the visual web interface in your browser
  requirements.txt            <- what to install (step 3 above)
  README.md                    <- this file
  HOW TO RUN AERIS.md           <- full no-coding-assumed instructions
  AERIS_AI_HANDOFF_PROMPT.md     <- paste this into another AI to give it
                                 full project context before asking for help
  aeris/
    ARCHITECTURE.md               <- technical details: how it's built, what's
                                 simplified, how to add new modules later
    core/                          <- shared physics + optimizer + CAD export engine
    modules/                       <- the airfoil / winglet / aircraft modules
    examples/                      <- the same three demos, runnable individually
    webapp/                        <- the local web server + browser interface
```
