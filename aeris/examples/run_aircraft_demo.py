"""
Demo: generate and optimize a full flying-wing shape (10 design variables:
span, root chord, taper, sweep, washout, root/tip camber, winglet height/
cant, operating alpha), the "Neuronautics-style" module built on top of the
same airfoil physics as Module 1 and the same winglet physics as Module 2.

Run: python -m aeris.examples.run_aircraft_demo
"""

from aeris.modules.aircraft_module import AircraftEvaluator
from aeris.core.optimizer import optimize
from aeris.core.logbook import Logbook


def main():
    evaluator = AircraftEvaluator(
        cruise_velocity_mps=10.0, altitude_m=0.0, target_CL=0.6,
    )
    print(evaluator.describe())
    print()

    result = optimize(evaluator, objective_fn=lambda m: m["L_D"],
                       maximize=True, popsize=30, maxiter=150, seed=7)

    print(result.summary())
    print(f"\nTotal designs evaluated: {len(result.history)}")

    log = Logbook("aircraft_logbook.json")
    design_vars = dict(zip(evaluator.var_names, result.x))
    run_id = log.log_run("aircraft", design_vars, result.metrics,
                          notes="First AERIS v1.0-style full flying-wing optimization")
    print(f"\nLogged as run_id = {run_id}")


if __name__ == "__main__":
    main()
