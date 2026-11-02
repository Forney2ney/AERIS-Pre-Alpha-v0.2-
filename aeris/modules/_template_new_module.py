"""
aeris.modules._template_new_module
=====================================
COPY THIS FILE to create any future module (propeller, rocket nozzle, spar
structure, etc). It is not imported by anything — it's a worked example of
the three things every new module needs to do:

  1. Inherit from Evaluator and declare var_names + bounds.
  2. Pull any physics it needs from aeris.core.knowledge_base (add new
     equations there if the physics doesn't exist yet — never re-derive
     physics inside a module file).
  3. Implement evaluate(x) -> dict, always including "_valid".

Once you've done that, aeris.core.optimizer.optimize() and
aeris.core.logbook.Logbook already work with it, unmodified.

--------------------------------------------------------------------------
EXAMPLE: sketch of a future propeller module (BEMT-based)
--------------------------------------------------------------------------

from aeris.core import knowledge_base as kb
from aeris.core.evaluator_base import Evaluator

class PropellerEvaluator(Evaluator):
    name = "propeller"
    var_names = ["diameter_m", "pitch_deg_root", "pitch_deg_tip", "n_blades"]
    bounds = [(0.15, 0.35), (10, 40), (5, 25), (2, 3)]

    def __init__(self, rpm, target_thrust_N, altitude_m=0.0):
        self.rpm = rpm
        self.target_thrust = target_thrust_N
        self.rho, self.mu, _, _ = kb.standard_atmosphere(altitude_m)

    def evaluate(self, x):
        diameter, pitch_root, pitch_tip, n_blades = x
        # ... call kb.blade_element_momentum_theory(...) once you've added
        #     that function to knowledge_base.py, exactly the way the
        #     aircraft module reuses kb.thin_airfoil_cl() ...
        thrust = 0.0     # placeholder
        power = 0.0      # placeholder
        efficiency = 0.0 # placeholder
        valid = True
        return {
            "thrust_N": thrust, "power_W": power,
            "efficiency": efficiency, "_valid": valid,
        }

--------------------------------------------------------------------------
Then, to run it, no other file needs to change:

    from aeris.modules.propeller_module import PropellerEvaluator
    from aeris.core.optimizer import optimize

    evaluator = PropellerEvaluator(rpm=8000, target_thrust_N=4.0)
    result = optimize(evaluator, objective_fn=lambda m: m["efficiency"])
--------------------------------------------------------------------------
"""
