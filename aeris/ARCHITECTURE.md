# AERIS — Core Engine (v0.1)

A modular computational aerospace engineering system: one shared physics
"knowledge base" + one generic optimizer + a growing set of pluggable
design modules. Built to be extended module-by-module (propeller, rocket
nozzle, spar/structure, etc.) without ever touching the core.

## Structure

```
aeris/
  core/
    knowledge_base.py    <- ALL physics/aero equations live here, and only here
    geometry.py           <- pure shape generation (NACA coords, planforms, tubercles, rotation)
    evaluator_base.py     <- the fixed contract every module must satisfy
    optimizer.py          <- ONE generic differential-evolution engine, evaluator-agnostic
                            (supports an optional on_generation callback for live progress)
    cad_export.py           <- DXF + STL writers, pure Python, no new dependency
    logbook.py             <- predicted-vs-measured "lab notebook" / feedback loop
  modules/
    airfoil_module.py       <- Module 1: airfoil + biomimicry (tubercles) +
                            export_airfoil_cad()
    winglet_module.py        <- Module 2: bio-inspired adaptive winglet optimizer
    aircraft_module.py       <- Module 3: full flying-wing generator (10 vars, Neuronautics-style)
                            + export_aircraft_cad()
    _template_new_module.py  <- copy this to add propeller/rocket-nozzle/etc later
  examples/
    run_airfoil_demo.py
    run_winglet_demo.py
    run_aircraft_demo.py
  webapp/
    server.py                <- local web server (Python standard library only)
    static/index.html, style.css, app.js  <- the browser interface
```

## Why it's built this way

- **One knowledge base.** Every equation (thin airfoil theory, skin friction,
  Reynolds number, tubercle effect, winglet effective-span model, Oswald
  efficiency...) lives in `core/knowledge_base.py`. Modules never
  re-derive physics — they call this file. That's what makes it "a system
  that knows all the physical concepts," not three separate scripts that
  happen to sit in the same folder.
- **One optimizer, evaluator-agnostic.** `core/optimizer.py` only knows
  about the `Evaluator` contract (`core/evaluator_base.py`). It has no idea
  whether it's optimizing an airfoil, a winglet, or a full aircraft — which
  is exactly what makes adding new modules cheap.
- **The aircraft module is built OUT OF the airfoil and winglet modules'
  physics**, not a separate re-implementation — `aircraft_module.py` calls
  `kb.thin_airfoil_cl(...)` and `kb.winglet_effective_AR(...)` directly,
  the same functions `airfoil_module.py` and `winglet_module.py` use. This
  mirrors the real engineering idea you described: airfoil design feeds
  into wing design, which is refined with biomimicry and winglets.
- **The logbook is the actual scientific engine of the project.** Every
  optimization run gets a `run_id`; when you later test a design physically,
  you attach the measurement to that same `run_id` and get an automatic
  prediction-vs-measurement error. This is the "predict → build → test →
  compare → improve the model" loop from your original AERIS concept,
  implemented as actual code rather than just a diagram.

## Running the demos

```bash
cd aeris/..              # the folder containing the aeris/ package
python3 -m aeris.examples.run_airfoil_demo
python3 -m aeris.examples.run_winglet_demo
python3 -m aeris.examples.run_aircraft_demo
```

Each demo optimizes a design, prints the result, and logs it to a
`*_logbook.json` file. Once you have real wind-tunnel or flight data, call:

```python
from aeris.core.logbook import Logbook
log = Logbook("airfoil_logbook.json")
log.log_test(run_id, {"CL": measured_CL, "CD": measured_CD})
log.print_summary()
```

## What's deliberately simplified (and needs your calibration)

Every one of these is flagged with a comment in `knowledge_base.py` — they
are the actual "science" of your Personal Project: predict with these,
measure for real, and correct the constants.

- `tubercle_effect()` — magnitude of biomimicry benefit is a placeholder
  tuned to look plausible, not derived from real tubercle experiments.
- `empirical_cl_max()` / `empirical_stall_alpha_deg()` — thin airfoil
  theory has no stall physics built in; these are simplified correlations
  standing in for real viscous/stall modeling.
- `winglet_effective_AR()` — a simplified span-efficiency model; real
  winglet benefit depends on load distribution and shape more precisely
  than this captures.
- `skin_friction_coefficient()` — a simple laminar/turbulent switch at a
  fixed Reynolds number; real transition depends on roughness and
  turbulence intensity.

## CAD export

`export_airfoil_cad()` (in `airfoil_module.py`) and `export_aircraft_cad()`
(in `aircraft_module.py`) turn a winning design's variables back into
geometry and write it to disk:

- Airfoil: a DXF 2D outline, plus an STL extruded solid (constant
  cross-section over a span you choose) — good for a wind-tunnel test
  piece.
- Aircraft: an STL loft built from the same spanwise stations the physics
  evaluation used (root to tip, including sweep offset and washout twist
  rotation) — so the printed shape matches exactly what was optimized.
  The winglet's aerodynamic benefit is included in the performance
  numbers, but its physical geometry is not yet in the exported solid —
  a good next addition.

Both writers are hand-rolled, pure-Python implementations of the DXF and
ASCII-STL text formats specifically so CAD export never requires a new pip
dependency.

## Local web interface

`aeris/webapp/server.py` is a small JSON API built entirely on Python's
standard library (`http.server.ThreadingHTTPServer`) — no Flask, no new
dependency. Each optimization run starts a background thread and returns a
`job_id` immediately; the frontend polls `/api/status/<job_id>` roughly
once a second, which is also how the live "optimization trace" chart gets
its data (the optimizer's `on_generation` callback appends to that job's
in-memory log every generation, so the frontend sees genuine live
progress, not a simulated one). The frontend (`webapp/static/`) is plain
HTML/CSS/JS, no build step, no framework.

## Adding a new module (e.g. propeller, rocket nozzle)

1. Copy `modules/_template_new_module.py` to `modules/your_module.py`.
2. Add any new physics equations you need to `core/knowledge_base.py`
   (don't put physics inside the module file).
3. Implement `evaluate(x) -> dict` with an `"_valid"` key.
4. Run it with the exact same `optimize()` and `Logbook` calls used by the
   three existing modules — nothing else needs to change.

## Next steps (matches the AERIS roadmap)

- v0.1 (this): airfoil + biomimicry, winglet, full aircraft generator — DONE.
- v0.2: physically test one optimized airfoil, log real CL/CD, calibrate
  `empirical_cl_max` / `tubercle_effect` against your own data.
- v0.3+: add a propeller module (`modules/_template_new_module.py` has a
  sketch), or a rocket-nozzle module, following the same pattern.
