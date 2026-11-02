"""
aeris.core.sensitivity
======================
Sensitivity analysis for understanding parameter influence on objectives.

Supports:
- Local sensitivity (finite differences)
- Global sensitivity (Morris method / elementary effects)
- Sobol indices (variance-based, when sample size permits)
"""

import numpy as np
from typing import Callable, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SensitivityResult:
    """Results of sensitivity analysis."""
    parameter_names: List[str]
    local_sensitivity: Dict[str, float]          # Finite difference gradients
    morris_mu: Optional[Dict[str, float]]        # Morris mean elementary effects
    morris_sigma: Optional[Dict[str, float]]     # Morris std of elementary effects
    sobol_first: Optional[Dict[str, float]]      # Sobol first-order indices
    sobol_total: Optional[Dict[str, float]]      # Sobol total-order indices
    base_point: np.ndarray
    base_metrics: Dict[str, Any]
    
    def summary(self, top_n: int = 10) -> str:
        lines = ["SENSITIVITY ANALYSIS SUMMARY", "=" * 40]
        
        if self.local_sensitivity:
            lines.append("\nLocal Sensitivity (finite difference at base point):")
            sorted_local = sorted(
                self.local_sensitivity.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            for name, val in sorted_local[:top_n]:
                lines.append(f"  {name:25s}: {val:+.6f}")
        
        if self.morris_mu is not None:
            lines.append("\nMorris Method (global screening):")
            sorted_mu = sorted(
                self.morris_mu.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for name, mu in sorted_mu[:top_n]:
                sigma = self.morris_sigma.get(name, 0)
                lines.append(f"  {name:25s}: μ*={mu:.6f}, σ={sigma:.6f}")
        
        if self.sobol_first is not None:
            lines.append("\nSobol Indices (variance-based):")
            sorted_sobol = sorted(
                self.sobol_first.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for name, s1 in sorted_sobol[:top_n]:
                st = self.sobol_total.get(name, 0)
                lines.append(f"  {name:25s}: S1={s1:.4f}, ST={st:.4f}")
        
        return "\n".join(lines)


def local_sensitivity(
    objective_fn: Callable[[np.ndarray], float],
    base_point: np.ndarray,
    bounds: List[Tuple[float, float]],
    step_fraction: float = 0.01,
) -> Dict[str, float]:
    """
    Compute local sensitivity via finite differences at a base point.
    
    Returns dict mapping parameter index -> sensitivity (df/dx * x/f for relative sensitivity).
    """
    n_vars = len(base_point)
    f0 = objective_fn(base_point)
    sensitivities = {}
    
    for i in range(n_vars):
        xl, xu = bounds[i]
        step = max(step_fraction * (xu - xl), 1e-8)
        
        # Forward difference
        x_plus = base_point.copy()
        x_plus[i] = min(base_point[i] + step, xu)
        f_plus = objective_fn(x_plus)
        
        # Backward difference
        x_minus = base_point.copy()
        x_minus[i] = max(base_point[i] - step, xl)
        f_minus = objective_fn(x_minus)
        
        # Central difference
        deriv = (f_plus - f_minus) / (2 * step)
        
        # Relative sensitivity: (df/dx) * (x / f)
        if abs(f0) > 1e-12 and abs(base_point[i]) > 1e-12:
            rel_sens = deriv * base_point[i] / f0
        else:
            rel_sens = deriv
        
        sensitivities[i] = rel_sens
    
    return sensitivities


def morris_sensitivity(
    objective_fn: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    n_trajectories: int = 10,
    n_levels: int = 4,
    step_ratio: float = None,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Morris method (elementary effects) for global sensitivity screening.
    
    Returns (mu_star, sigma) for each parameter.
    mu_star = mean of absolute elementary effects (overall influence)
    sigma = std of elementary effects (non-linearity/interactions)
    """
    n_vars = len(bounds)
    if step_ratio is None:
        step_ratio = 1.0 / (n_levels - 1)
    
    delta = step_ratio
    mu_star = {i: [] for i in range(n_vars)}
    
    for _ in range(n_trajectories):
        # Random base point in grid
        base = np.zeros(n_vars)
        for i in range(n_vars):
            xl, xu = bounds[i]
            grid_points = np.linspace(xl, xu, n_levels)
            base[i] = np.random.choice(grid_points)
        
        # Random permutation of parameters
        perm = np.random.permutation(n_vars)
        
        x_current = base.copy()
        f_base = objective_fn(x_current)
        
        for param_idx in perm:
            x_next = x_current.copy()
            xl, xu = bounds[param_idx]
            
            # Move in random direction
            step = delta * (xu - xl)
            if np.random.random() < 0.5:
                x_next[param_idx] = min(x_current[param_idx] + step, xu)
            else:
                x_next[param_idx] = max(x_current[param_idx] - step, xl)
            
            f_next = objective_fn(x_next)
            
            # Elementary effect
            ee = (f_next - f_base) / (step * (xu - xl) / (xu - xl)) if abs(step) > 1e-12 else 0
            mu_star[param_idx].append(abs(ee))
            
            x_current = x_next
            f_base = f_next
    
    # Compute statistics
    mu_star_final = {i: np.mean(vals) if vals else 0.0 for i, vals in mu_star.items()}
    sigma_final = {i: np.std(vals) if len(vals) > 1 else 0.0 for i, vals in mu_star.items()}
    
    return mu_star_final, sigma_final


def sobol_sensitivity(
    objective_fn: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    n_samples: int = 1024,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Sobol sensitivity indices using Saltelli's sampling scheme.
    
    Returns (first_order, total_order) indices for each parameter.
    Requires n_samples * (n_vars + 2) evaluations.
    """
    n_vars = len(bounds)
    
    # Generate base samples using Sobol sequence (using random for now)
    # For production, use scipy.stats.qmc.Sobol
    n_total = n_samples * (n_vars + 2)
    
    A = np.zeros((n_samples, n_vars))
    B = np.zeros((n_samples, n_vars))
    for i in range(n_vars):
        xl, xu = bounds[i]
        A[:, i] = np.random.uniform(xl, xu, n_samples)
        B[:, i] = np.random.uniform(xl, xu, n_samples)
    
    # Evaluate
    f_A = np.array([objective_fn(A[i]) for i in range(n_samples)])
    f_B = np.array([objective_fn(B[i]) for i in range(n_samples)])
    
    # Create C matrices (A with column i from B)
    f_C = np.zeros((n_vars, n_samples))
    for i in range(n_vars):
        C = A.copy()
        C[:, i] = B[:, i]
        f_C[i] = np.array([objective_fn(C[j]) for j in range(n_samples)])
    
    # Compute indices
    var_y = np.var(np.concatenate([f_A, f_B]))
    
    first_order = {}
    total_order = {}
    
    for i in range(n_vars):
        # First order: V[E[Y|X_i]] / V[Y]
        first_order[i] = np.mean(f_B * (f_C[i] - f_A)) / var_y if var_y > 0 else 0
        
        # Total order: E[V[Y|X_{~i}]] / V[Y] = 1 - V[E[Y|X_{~i}]]/V[Y]
        total_order[i] = np.mean((f_A - f_C[i])**2) / (2 * var_y) if var_y > 0 else 0
    
    return first_order, total_order


def analyze_sensitivity(
    evaluator,
    base_design: np.ndarray,
    objective_fn: Callable[[Dict], float],
    method: str = "local",
    **kwargs
) -> "SensitivityResult":
    """
    High-level sensitivity analysis wrapper.
    
    Args:
        evaluator: Evaluator instance (for bounds, var_names)
        base_design: Base design point (best design from optimization)
        objective_fn: Function(metrics) -> scalar objective
        method: "local", "morris", "sobol", or "all"
        
    Returns:
        SensitivityResult with all computed sensitivities
    """
    # Wrap evaluator for objective function
    def wrapped_obj(x):
        metrics = evaluator.evaluate(x)
        return objective_fn(metrics)
    
    bounds = evaluator.bounds
    var_names = evaluator.var_names
    base_metrics = evaluator.evaluate(base_design)
    
    local_sens = {}
    morris_mu = None
    morris_sigma = None
    sobol_first = None
    sobol_total = None
    
    if method in ("local", "all"):
        sens = local_sensitivity(wrapped_obj, base_design, bounds, **kwargs.get("local", {}))
        local_sens = {var_names[i]: v for i, v in sens.items()}
    
    if method in ("morris", "all"):
        mu, sigma = morris_sensitivity(wrapped_obj, bounds, **kwargs.get("morris", {}))
        morris_mu = {var_names[i]: v for i, v in mu.items()}
        morris_sigma = {var_names[i]: v for i, v in sigma.items()}
    
    if method in ("sobol", "all"):
        s1, st = sobol_sensitivity(wrapped_obj, bounds, **kwargs.get("sobol", {}))
        sobol_first = {var_names[i]: v for i, v in s1.items()}
        sobol_total = {var_names[i]: v for i, v in st.items()}
    
    return SensitivityResult(
        parameter_names=var_names,
        local_sensitivity=local_sens,
        morris_mu=morris_mu,
        morris_sigma=morris_sigma,
        sobol_first=sobol_first,
        sobol_total=sobol_total,
        base_point=base_design.copy(),
        base_metrics=base_metrics,
    )