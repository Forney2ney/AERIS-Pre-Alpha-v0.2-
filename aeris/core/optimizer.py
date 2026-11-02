"""
aeris.core.optimizer
=====================
One generic optimizer engine used by EVERY module. It only ever talks to the
Evaluator contract (aeris.core.evaluator_base.Evaluator), never to
module-specific physics. Adding a new project module never requires
touching this file.

Supports both single-objective (Differential Evolution) and multi-objective
(NSGA-II) optimization.
"""

import numpy as np
from scipy.optimize import differential_evolution
from typing import Callable, List, Dict, Any, Union, Optional


class OptimizationResult:
    def __init__(self, x, metrics, score, history, generation_log, success):
        self.x = x
        self.metrics = metrics
        self.score = score
        self.history = history            # every single evaluation tried
        self.generation_log = generation_log  # best-so-far per generation
        self.success = success

    def summary(self) -> str:
        lines = [f"Best score: {self.score:.4f}", "Best design variables:"]
        for k, v in zip(self.metrics.get("_var_names", []), self.x):
            lines.append(f"  {k:22s} = {v:.5f}")
        lines.append("Metrics:")
        for k, v in self.metrics.items():
            if k.startswith("_"):
                continue
            lines.append(f"  {k:22s} = {v}")
        return "\n".join(lines)


def optimize(evaluator, objective_fn, maximize=True, popsize=20, maxiter=100,
             seed=None, workers=1, penalty=1e6, on_generation=None,
             multi_objective=False):
    """Run optimization against any Evaluator.

    evaluator     : an instance satisfying the Evaluator contract
    objective_fn  : function(metrics_dict) -> float, the scalar to
                    maximize/minimize (or list of functions for multi-objective)
    maximize      : if True, objective_fn is maximized; else minimized
                    (or list of bools for multi-objective)
    multi_objective: if True, use NSGA-II multi-objective optimization
    on_generation : optional callable(generation_index, metrics_dict, score)
                    called once per generation with the best-so-far result —
                    used by the web UI to stream live progress while a run
                    is still in progress. Purely optional; every existing
                    caller (CLI demos, start_here.py) is unaffected.
    """
    if multi_objective:
        from aeris.core.optimizer_mo import optimize as mo_optimize
        return mo_optimize(
            evaluator=evaluator,
            objective_fn=objective_fn,
            maximize=maximize,
            popsize=popsize,
            maxiter=maxiter,
            seed=seed,
            workers=workers,
            penalty=penalty,
            on_generation=on_generation,
            multi_objective=True,
        )
    
    history = []
    generation_log = []
    gen_counter = {"n": 0}

    def scored(x):
        metrics = evaluator.evaluate(x)
        valid = metrics.get("_valid", True)
        raw_score = objective_fn(metrics) if valid else -np.inf
        history.append({"x": list(map(float, x)), "metrics": metrics,
                         "score": raw_score, "valid": valid})
        if not valid:
            return penalty
        return -raw_score if maximize else raw_score

    def callback(xk, convergence):
        metrics = evaluator.evaluate(xk)
        score = objective_fn(metrics)
        gen_counter["n"] += 1
        generation_log.append({
            "x": list(map(float, xk)),
            "metrics": metrics,
            "convergence": float(convergence),
        })
        if on_generation is not None:
            on_generation(gen_counter["n"], metrics, score)

    result = differential_evolution(
        scored, evaluator.bounds, popsize=popsize, maxiter=maxiter,
        seed=seed, workers=workers, polish=True, callback=callback,
        updating="deferred" if workers != 1 else "immediate",
    )

    best_metrics = evaluator.evaluate(result.x)
    best_metrics["_var_names"] = evaluator.var_names
    best_score = objective_fn(best_metrics)

    return OptimizationResult(
        x=result.x, metrics=best_metrics, score=best_score,
        history=history, generation_log=generation_log,
        success=result.success,
    )

# Re-export multi-objective components
__all__ = ["OptimizationResult", "optimize", "NSGA2Optimizer", "MOOptimizationResult"]
