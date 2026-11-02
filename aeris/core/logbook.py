"""
aeris.core.logbook
===================
The generic "lab notebook." Every module logs its optimization runs here,
and later logs real physical test results against the same run ID. This is
what turns AERIS from "a shape generator" into "a computational model you
are actually validating" — the core scientific claim of the whole project.
Evaluator-agnostic: works identically for airfoil, winglet, aircraft, or any
future module.
"""

import json
import os
import time
import uuid
from typing import Any
import numpy as np


def _to_json_serializable(obj: Any) -> Any:
    """Convert numpy types and other non-JSON-serializable objects to JSON-serializable types."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    else:
        return obj


class Logbook:
    def __init__(self, path="aeris_logbook.json"):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump([], f)

    def _load(self):
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except json.JSONDecodeError:
            # Corrupted file, return empty list
            return []

    def _save(self, data):
        # Convert all data to JSON-serializable types before saving
        serializable_data = _to_json_serializable(data)
        # Write to temp file first, then atomic rename
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=os.path.dirname(self.path) or '.', suffix='.tmp') as tf:
            json.dump(serializable_data, tf, indent=2)
            tf.flush()
            os.fsync(tf.fileno())
            temp_path = tf.name
        # Atomic rename
        os.replace(temp_path, self.path)

    def log_run(self, module: str, design_vars: dict, predicted_metrics: dict,
                notes: str = "") -> str:
        """Record a fresh optimization/design result. Returns a run_id you
        use later to attach real test data."""
        run_id = str(uuid.uuid4())[:8]
        entry = {
            "run_id": run_id,
            "module": module,
            "timestamp": time.time(),
            "design_vars": _to_json_serializable(design_vars),
            "predicted_metrics": _to_json_serializable(predicted_metrics),
            "measured_metrics": None,
            "error": None,
            "notes": notes,
        }
        data = self._load()
        data.append(entry)
        self._save(data)
        return run_id

    def log_test(self, run_id: str, measured_metrics: dict, notes: str = ""):
        """Attach real physical test results to a previously logged run and
        automatically compute the prediction error."""
        data = self._load()
        for entry in data:
            if entry["run_id"] == run_id:
                entry["measured_metrics"] = _to_json_serializable(measured_metrics)
                entry["error"] = self.compute_error(
                    entry["predicted_metrics"], measured_metrics)
                entry["test_notes"] = notes
                self._save(data)
                return entry
        raise KeyError(f"No run with id {run_id} found in {self.path}")

    @staticmethod
    def compute_error(predicted: dict, measured: dict) -> dict:
        """Relative error per numeric key that exists in both dicts."""
        error = {}
        for k, meas_v in measured.items():
            pred_v = predicted.get(k)
            if pred_v is None or not isinstance(meas_v, (int, float)):
                continue
            if meas_v == 0:
                error[k] = None
            else:
                error[k] = abs(pred_v - meas_v) / abs(meas_v)
        return error

    def history(self, module: str = None):
        data = self._load()
        if module:
            return [e for e in data if e["module"] == module]
        return data

    def print_summary(self, module: str = None):
        for entry in self.history(module):
            print(f"[{entry['run_id']}] {entry['module']} "
                  f"({time.strftime('%Y-%m-%d', time.localtime(entry['timestamp']))})")
            print(f"  predicted: {entry['predicted_metrics']}")
            if entry["measured_metrics"]:
                print(f"  measured:  {entry['measured_metrics']}")
                print(f"  error:     {entry['error']}")
            else:
                print("  measured:  (not yet tested)")
