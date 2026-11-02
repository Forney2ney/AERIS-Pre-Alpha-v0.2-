"""
Demo: optimize an airfoil (with biomimicry tubercles enabled) for a target
flight condition, log the run, and show how you'd later attach real test
data.

Run: python -m aeris.examples.run_airfoil_demo
"""

from aeris.modules.airfoil_module import AirfoilEvaluator
from aeris.core.optimizer import optimize
from aeris.core.logbook import Logbook


def main():
    evaluator = AirfoilEvaluator(
        alpha_deg=6.0, velocity_mps=12.0, chord_m=0.15, altitude_m=0.0,
        use_biomimicry=True,
    )
    print(evaluator.describe())
    print()

    # objective: maximize L/D, i.e. best glide efficiency at this condition
    result = optimize(evaluator, objective_fn=lambda m: m["L_D"],
                       maximize=True, popsize=20, maxiter=60, seed=42)

    print(result.summary())
    print(f"\nTotal designs evaluated: {len(result.history)}")
    print(f"Generations logged: {len(result.generation_log)}")

    # log this run so it can be compared against real wind-tunnel data later
    log = Logbook("airfoil_logbook.json")
    design_vars = dict(zip(evaluator.var_names, result.x))
    run_id = log.log_run("airfoil", design_vars, result.metrics,
                          notes="First AERIS v0.1 airfoil optimization run")
    print(f"\nLogged as run_id = {run_id}")
    print("Once you test this airfoil in your wind tunnel, call:")
    print(f"  log.log_test('{run_id}', {{'CL': <measured_CL>, 'CD': <measured_CD>}})")


if __name__ == "__main__":
    main()
