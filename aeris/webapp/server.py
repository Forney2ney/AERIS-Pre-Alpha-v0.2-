"""
aeris.webapp.server
=====================
A tiny local web server for AERIS, built entirely on Python's standard
library (http.server) so running the web UI never requires installing
anything beyond requirements.txt (numpy, scipy). It exposes a small JSON
API that the static frontend (aeris/webapp/static/) talks to, and runs each
optimization job in a background thread so the page stays responsive while
a run (especially the full-aircraft module) is in progress.

This file only ever talks to the same Evaluator / optimize() / Logbook
building blocks used by start_here.py and the CLI examples — it is a new
front door onto the same core engine, not a separate implementation.
"""

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from aeris.core.optimizer import optimize
from aeris.core.logbook import Logbook
from aeris.core.model_validity import ModelValidityEngine
from aeris.modules.airfoil_module import AirfoilEvaluator, export_airfoil_cad
from aeris.modules.winglet_module import WingletEvaluator
from aeris.modules.aircraft_module import AircraftEvaluator, export_aircraft_cad

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
LOGBOOK_PATH = "aeris_logbook.json"       # shared with start_here.py
EXPORT_DIR = "aeris_cad_exports"

JOBS = {}            # job_id -> dict, see start_job() for shape
JOBS_LOCK = threading.Lock()

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json",
}


# ---------------------------------------------------------------------------
# Job execution (runs in a background thread per request)
# ---------------------------------------------------------------------------

def _run_airfoil_job(job_id, params):
    ev = AirfoilEvaluator(
        alpha_deg=params.get("alpha_deg", 6.0),
        velocity_mps=params.get("velocity_mps", 12.0),
        chord_m=params.get("chord_m", 0.15),
        altitude_m=params.get("altitude_m", 0.0),
        use_biomimicry=params.get("use_biomimicry", True),
    )

    def on_gen(gen, metrics, score):
        with JOBS_LOCK:
            JOBS[job_id]["live_log"].append({"gen": gen, "score": score,
                                              "metrics": metrics})

    result = optimize(ev, objective_fn=lambda m: m["L_D"], maximize=True,
                       popsize=20, maxiter=60, seed=42, on_generation=on_gen)
    design_vars = dict(zip(ev.var_names, result.x))

    # Get polar data for the best design
    polar_metrics = ev.evaluate(result.x, return_polar=True)
    
    # Merge polar data into result metrics
    final_metrics = dict(result.metrics)
    if "polar" in polar_metrics:
        final_metrics["polar"] = polar_metrics["polar"]
        final_metrics["baseline_polar"] = polar_metrics.get("baseline_polar")
        final_metrics["polar_source"] = polar_metrics.get("polar_source", "analytical")

    log = Logbook(LOGBOOK_PATH)
    run_id = log.log_run("airfoil", design_vars, final_metrics,
                          notes="Run from web UI")

    with JOBS_LOCK:
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["run_id"] = run_id
        JOBS[job_id]["design_vars"] = design_vars
        JOBS[job_id]["metrics"] = final_metrics
        JOBS[job_id]["chord_m"] = params.get("chord_m", 0.15)
        JOBS[job_id]["span_m"] = params.get("span_m", 0.3)


def _run_winglet_job(job_id, params):
    ev = WingletEvaluator(
        base_span_m=params.get("base_span_m", 1.2),
        base_area_m2=params.get("base_area_m2", 0.18),
        operating_CL=params.get("operating_CL", 0.55),
        base_CD0=params.get("base_CD0", 0.02),
    )

    def on_gen(gen, metrics, score):
        with JOBS_LOCK:
            JOBS[job_id]["live_log"].append({"gen": gen, "score": score,
                                              "metrics": metrics})

    result = optimize(ev, objective_fn=lambda m: m["L_D"], maximize=True,
                       popsize=20, maxiter=60, seed=1, on_generation=on_gen)
    design_vars = dict(zip(ev.var_names, result.x))

    log = Logbook(LOGBOOK_PATH)
    run_id = log.log_run("winglet", design_vars, result.metrics,
                          notes="Run from web UI")

    with JOBS_LOCK:
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["run_id"] = run_id
        JOBS[job_id]["design_vars"] = design_vars
        JOBS[job_id]["metrics"] = result.metrics


def _run_aircraft_job(job_id, params):
    ev = AircraftEvaluator(
        cruise_velocity_mps=params.get("cruise_velocity_mps", 10.0),
        altitude_m=params.get("altitude_m", 0.0),
        target_CL=params.get("target_CL", 0.6),
    )

    def on_gen(gen, metrics, score):
        with JOBS_LOCK:
            JOBS[job_id]["live_log"].append({"gen": gen, "score": score,
                                              "metrics": metrics})

    result = optimize(ev, objective_fn=lambda m: m["L_D"], maximize=True,
                       popsize=30, maxiter=150, seed=7, on_generation=on_gen)
    design_vars = dict(zip(ev.var_names, result.x))

    log = Logbook(LOGBOOK_PATH)
    run_id = log.log_run("aircraft", design_vars, result.metrics,
                          notes="Run from web UI")

    with JOBS_LOCK:
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["run_id"] = run_id
        JOBS[job_id]["design_vars"] = design_vars
        JOBS[job_id]["metrics"] = result.metrics


JOB_RUNNERS = {
    "airfoil": _run_airfoil_job,
    "winglet": _run_winglet_job,
    "aircraft": _run_aircraft_job,
}


def start_job(module, params):
    job_id = str(uuid.uuid4())[:8]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "running", "module": module, "started": time.time(),
            "live_log": [], "run_id": None, "design_vars": None,
            "metrics": None, "error": None,
        }

    def target():
        try:
            JOB_RUNNERS[module](job_id, params)
        except Exception as exc:  # surface errors to the UI instead of hanging
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["error"] = str(exc)

    threading.Thread(target=target, daemon=True).start()
    return job_id


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class AerisHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet; errors still surface via /api routes

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_download(self, path, filename):
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    # ---- routing ----

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/":
            return self._send_file(os.path.join(STATIC_DIR, "index.html"),
                                    CONTENT_TYPES[".html"])

        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            full = os.path.join(STATIC_DIR, rel)
            ext = os.path.splitext(full)[1]
            if os.path.isfile(full) and ext in CONTENT_TYPES:
                return self._send_file(full, CONTENT_TYPES[ext])
            return self._send_json({"error": "not found"}, 404)

        if path.startswith("/api/status/"):
            job_id = path[len("/api/status/"):]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is None:
                    return self._send_json({"error": "unknown job_id"}, 404)
                return self._send_json(dict(job))

        if path == "/api/logbook":
            log = Logbook(LOGBOOK_PATH)
            return self._send_json({"entries": log.history()})

        if path.startswith("/api/physics_health/"):
            job_id = path[len("/api/physics_health/"):]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None or job["status"] != "done":
                return self._send_json({"error": "job not ready"}, 404)
            
            evaluator = None
            if job["module"] == "airfoil":
                evaluator = AirfoilEvaluator(
                    alpha_deg=job.get("alpha_deg", 6.0),
                    velocity_mps=job.get("velocity_mps", 12.0),
                    chord_m=job.get("chord_m", 0.15),
                    altitude_m=job.get("altitude_m", 0.0),
                    use_biomimicry=job.get("use_biomimicry", True),
                )
            elif job["module"] == "winglet":
                evaluator = WingletEvaluator(
                    base_span_m=job.get("base_span_m", 1.2),
                    base_area_m2=job.get("base_area_m2", 0.18),
                    operating_CL=job.get("operating_CL", 0.55),
                    base_CD0=job.get("base_CD0", 0.02),
                )
            elif job["module"] == "aircraft":
                evaluator = AircraftEvaluator(
                    cruise_velocity_mps=job.get("cruise_velocity_mps", 10.0),
                    altitude_m=job.get("altitude_m", 0.0),
                    target_CL=job.get("target_CL", 0.6),
                )
            
            if evaluator:
                conditions = evaluator.get_model_validity_conditions()
                # Add stall_alpha_deg from metrics if available
                metrics = job.get("metrics", {})
                if "stall_AoA_deg" in metrics:
                    conditions["stall_alpha_deg"] = metrics["stall_AoA_deg"]
                elif "tip_stall_margin" in metrics:
                    # Aircraft doesn't have single stall angle
                    pass
                
                engine = ModelValidityEngine()
                checks = engine.check_all(conditions)
                overall = engine.overall_status(checks)
                return self._send_json({
                    "overall": overall.value,
                    "checks": [c.to_dict() for c in checks],
                    "summary": engine.summary(checks),
                })
            else:
                return self._send_json({"error": "unknown module"}, 400)

        if path.startswith("/api/export/"):
            parts = path[len("/api/export/"):].split("/")
            if len(parts) != 2:
                return self._send_json({"error": "bad export path"}, 400)
            job_id, filetype = parts
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None or job["status"] != "done":
                return self._send_json({"error": "job not ready"}, 404)

            os.makedirs(EXPORT_DIR, exist_ok=True)
            try:
                if job["module"] == "airfoil":
                    paths = export_airfoil_cad(
                        job["design_vars"], chord_m=job.get("chord_m", 0.15),
                        span_m=job.get("span_m", 0.3),
                        out_dir=EXPORT_DIR, base_name=f"airfoil_{job_id}")
                elif job["module"] == "aircraft":
                    paths = export_aircraft_cad(
                        job["design_vars"], out_dir=EXPORT_DIR,
                        base_name=f"aircraft_{job_id}")
                else:
                    return self._send_json(
                        {"error": "this module has no CAD export yet"}, 400)
            except Exception as exc:
                return self._send_json({"error": f"export failed: {exc}"}, 500)

            if filetype not in paths:
                return self._send_json(
                    {"error": f"no {filetype} export for this module"}, 400)
            fname = os.path.basename(paths[filetype])
            return self._send_download(paths[filetype], fname)

        return self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        params = self._read_json_body()

        if path.startswith("/api/run/"):
            module = path[len("/api/run/"):]
            if module not in JOB_RUNNERS:
                return self._send_json({"error": f"unknown module {module}"}, 400)
            job_id = start_job(module, params)
            return self._send_json({"job_id": job_id})

        if path == "/api/test_result":
            run_id = params.get("run_id")
            measured = params.get("measured", {})
            if not run_id or not measured:
                return self._send_json(
                    {"error": "run_id and measured are required"}, 400)
            log = Logbook(LOGBOOK_PATH)
            try:
                entry = log.log_test(run_id, measured)
            except KeyError as exc:
                return self._send_json({"error": str(exc)}, 404)
            return self._send_json(entry)

        return self._send_json({"error": "not found"}, 404)


def run(port=8765):
    server = ThreadingHTTPServer(("127.0.0.1", port), AerisHandler)
    print(f"AERIS web UI running at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping AERIS web server.")
        server.shutdown()


if __name__ == "__main__":
    run()
