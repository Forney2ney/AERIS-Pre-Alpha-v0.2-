"""
Demo: optimize an adaptive winglet for a fixed baseline wing.

Run: python -m aeris.examples.run_winglet_demo
"""

from aeris.modules.winglet_module import WingletEvaluator
from aeris.core.optimizer import optimize


def main():
    evaluator = WingletEvaluator(
        base_span_m=1.2, base_area_m2=0.18, operating_CL=0.55, base_CD0=0.02,
    )
    print(evaluator.describe())
    print()

    result = optimize(evaluator, objective_fn=lambda m: m["L_D"],
                       maximize=True, popsize=20, maxiter=60, seed=1)

    print(result.summary())
    print(f"\nEfficiency improvement over no-winglet baseline: "
          f"{result.metrics['L_D_improvement_pct']:.2f}%")


if __name__ == "__main__":
    main()
