"""
aeris.core.explain
===================
Explainable optimization output - understanding WHY a design was chosen.

Generates human-readable explanations of optimization decisions based on
actual numerical changes in the design space and objective landscape.
"""

import numpy as np
from typing import Callable, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from aeris.core.sensitivity import analyze_sensitivity, SensitivityResult


@dataclass
class DesignExplanation:
    """Explanation for a specific design choice."""
    design_name: str
    design_variables: Dict[str, float]
    objectives: Dict[str, float]
    baseline_comparison: Optional[Dict[str, Any]] = None
    sensitivity: Optional[SensitivityResult] = None
    key_changes: List[str] = field(default_factory=list)
    tradeoffs: List[str] = field(default_factory=list)
    constraint_status: Dict[str, bool] = field(default_factory=dict)
    
    def to_text(self) -> str:
        """Generate human-readable explanation."""
        lines = [
            f"DESIGN EXPLANATION: {self.design_name}",
            "=" * 50,
            "",
            "DESIGN VARIABLES:",
        ]
        
        for k, v in self.design_variables.items():
            lines.append(f"  {k:25s} = {v:.5f}")
        
        lines.append("\nOBJECTIVES:")
        for k, v in self.objectives.items():
            lines.append(f"  {k:25s} = {v:.5f}")
        
        if self.baseline_comparison:
            lines.append("\nVS. BASELINE:")
            for k, v in self.baseline_comparison.items():
                if isinstance(v, dict) and "improvement_pct" in v:
                    direction = "improved" if v["improvement_pct"] > 0 else "worsened"
                    lines.append(f"  {k:25s}: {direction} by {abs(v['improvement_pct']):.1f}%")
        
        if self.sensitivity and self.sensitivity.local_sensitivity:
            lines.append("\nKEY SENSITIVITIES (at this design):")
            sorted_sens = sorted(
                self.sensitivity.local_sensitivity.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            for name, val in list(sorted_sens)[:5]:
                lines.append(f"  {name:25s}: {val:+.4f} (relative)")
        
        if self.key_changes:
            lines.append("\nKEY DESIGN DECISIONS:")
            for change in self.key_changes:
                lines.append(f"  - {change}")
        
        if self.tradeoffs:
            lines.append("\nTRADEOFFS:")
            for tradeoff in self.tradeoffs:
                lines.append(f"  - {tradeoff}")
        
        if self.constraint_status:
            lines.append("\nCONSTRAINT STATUS:")
            for name, satisfied in self.constraint_status.items():
                status = "SATISFIED" if satisfied else "VIOLATED"
                lines.append(f"  {name:25s}: {status}")
        
        return "\n".join(lines)


def explain_optimization(
    evaluator,
    optimal_design: np.ndarray,
    objective_fn: Callable[[Dict], float],
    baseline_design: Optional[np.ndarray] = None,
    constraint_set=None,
    sensitivity_method: str = "local",
    var_names: Optional[List[str]] = None,
) -> DesignExplanation:
    """
    Generate a comprehensive explanation of why this design was chosen.
    
    Args:
        evaluator: Evaluator instance
        optimal_design: The optimal design variables (x)
        objective_fn: Objective function (metrics -> scalar)
        baseline_design: Optional baseline design for comparison
        constraint_set: ConstraintSet for constraint status
        sensitivity_method: Sensitivity analysis method
        
    Returns:
        DesignExplanation with human-readable explanation
    """
    var_names = var_names or evaluator.var_names
    design_vars = {var_names[i]: float(optimal_design[i]) for i in range(len(optimal_design))}
    optimal_metrics = evaluator.evaluate(optimal_design)
    
    objectives = {"objective": objective_fn(optimal_metrics)}
    
    # Add relevant metrics
    for k in ["L_D", "CL", "CD", "CL_max", "stall_AoA_deg", "best_LD", "best_LD_alpha"]:
        if k in optimal_metrics:
            objectives[k] = optimal_metrics[k]
    
    # Baseline comparison
    baseline_comparison = None
    if baseline_design is not None:
        baseline_metrics = evaluator.evaluate(baseline_design)
        baseline_obj = objective_fn(baseline_metrics)
        opt_obj = objectives["objective"]
        
        baseline_comparison = {}
        for k in ["L_D", "CL", "CD", "CL_max"]:
            if k in optimal_metrics and k in baseline_metrics:
                opt_val = optimal_metrics[k]
                base_val = baseline_metrics[k]
                if base_val != 0:
                    pct = 100 * (opt_val - base_val) / abs(base_val)
                    baseline_comparison[k] = {
                        "optimal": opt_val,
                        "baseline": base_val,
                        "improvement": opt_val - base_val,
                        "improvement_pct": pct,
                    }
    
    # Sensitivity analysis
    sensitivity = analyze_sensitivity(
        evaluator, optimal_design, objective_fn, 
        method=sensitivity_method
    )
    
    # Generate key insights
    key_changes = []
    tradeoffs = []
    
    if sensitivity and sensitivity.local_sensitivity:
        # Find most influential parameters
        sorted_sens = sorted(
            sensitivity.local_sensitivity.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        for name, val in sorted_sens[:3]:
            if abs(val) > 0.1:
                direction = "increasing" if val > 0 else "decreasing"
                key_changes.append(
                    f"{name} has high influence ({val:+.3f}); {direction} it improves the objective"
                )
    
    # Check for tradeoffs (constraints near limits)
    if "constraints" in optimal_metrics:
        for name, c in optimal_metrics["constraints"].items():
            if isinstance(c, dict) and "margin" in c:
                margin = c.get("margin", 0)
                if abs(margin) < 0.1:  # Near constraint boundary
                    tradeoffs.append(
                        f"{name} is near its limit (margin={margin:.4f})"
                    )
    
    # Check for objective tradeoffs
    if "L_D" in objectives and "CL" in objectives:
        # Usually L/D and CL can trade off
        pass
    
    # Constraint status
    constraint_status = {}
    if constraint_set:
        results = constraint_set.evaluate(optimal_metrics)
        for r in results:
            constraint_status[r.name] = r.satisfied
    elif "constraints" in optimal_metrics:
        for name, c in optimal_metrics["constraints"].items():
            if isinstance(c, dict):
                constraint_status[name] = c.get("satisfied", True)
    
    return DesignExplanation(
        design_name="Optimized Design",
        design_variables=design_vars,
        objectives=objectives,
        baseline_comparison=baseline_comparison,
        sensitivity=sensitivity,
        key_changes=key_changes,
        tradeoffs=tradeoffs,
        constraint_status=constraint_status,
    )


def explain_pareto_front(
    evaluator,
    pareto_x: np.ndarray,
    pareto_metrics: List[Dict],
    pareto_scores: np.ndarray,
    objective_names: List[str],
    objective_fns: List[Callable],
    baseline_design: Optional[np.ndarray] = None,
) -> List[DesignExplanation]:
    """
    Generate explanations for all Pareto-optimal designs.
    
    Returns list of DesignExplanation sorted by first objective.
    """
    explanations = []
    
    for i, (x, metrics, scores) in enumerate(zip(pareto_x, pareto_metrics, pareto_scores)):
        objectives = {}
        for j, name in enumerate(objective_names):
            objectives[name] = scores[j]
        
        # Add relevant metrics
        for k in ["L_D", "CL", "CD", "CL_max", "stall_AoA_deg"]:
            if k in metrics:
                objectives[k] = metrics[k]
        
        # Compare to baseline
        baseline_comparison = None
        if baseline_design is not None:
            baseline_metrics = evaluator.evaluate(baseline_design)
            baseline_comparison = {}
            for k in ["L_D", "CL", "CD", "CL_max"]:
                if k in metrics and k in baseline_metrics:
                    opt_val = metrics[k]
                    base_val = baseline_metrics[k]
                    if base_val != 0:
                        pct = 100 * (opt_val - base_val) / abs(base_val)
                        baseline_comparison[k] = {
                            "optimal": opt_val,
                            "baseline": base_val,
                            "improvement_pct": pct,
                        }
        
        # Sensitivity for this design
        sensitivity = None
        if objective_fns:
            sensitivity = analyze_sensitivity(
                evaluator, x, objective_fns[0], method="local"
            )
        
        # Rank-based explanation
        key_changes = []
        if i == 0:
            key_changes.append("Pareto-optimal: best on first objective")
        else:
            key_changes.append(f"Pareto rank {i+1}: tradeoff between {objective_names}")
        
        explanations.append(DesignExplanation(
            design_name=f"Pareto Rank {i+1}",
            design_variables={evaluator.var_names[k]: float(x[k]) for k in range(len(x))},
            objectives=objectives,
            baseline_comparison=baseline_comparison,
            sensitivity=sensitivity,
            key_changes=key_changes,
        ))
    
    return explanations


def compare_designs(
    evaluator,
    design_a: np.ndarray,
    design_b: np.ndarray,
    name_a: str = "Design A",
    name_b: str = "Design B",
    objective_fn: Callable[[Dict], float] = None,
) -> str:
    """Generate a comparison report between two designs."""
    metrics_a = evaluator.evaluate(design_a)
    metrics_b = evaluator.evaluate(design_b)
    
    lines = [
        f"DESIGN COMPARISON: {name_a} vs {name_b}",
        "=" * 50,
        "",
        "DESIGN VARIABLES:",
    ]
    
    var_names = evaluator.var_names
    for i, name in enumerate(var_names):
        diff = design_a[i] - design_b[i]
        lines.append(f"  {name:25s}: A={design_a[i]:.5f}, B={design_b[i]:.5f}, Δ={diff:+.5f}")
    
    lines.append("\nMETRICS:")
    for k in ["CL", "CD", "L_D", "CL_max", "stall_AoA_deg", "best_LD", "best_LD_alpha", "CD0"]:
        if k in metrics_a and k in metrics_b:
            diff = metrics_a[k] - metrics_b[k]
            lines.append(f"  {k:25s}: A={metrics_a[k]:.5f}, B={metrics_b[k]:.5f}, Δ={diff:+.5f}")
    
    if objective_fn:
        obj_a = objective_fn(metrics_a)
        obj_b = objective_fn(metrics_b)
        lines.append(f"\nOBJECTIVE: A={obj_a:.5f}, B={obj_b:.5f}, Δ={obj_a-obj_b:+.5f}")
    
    return "\n".join(lines)


def generate_design_report(
    evaluator,
    optimal_design: np.ndarray,
    pareto_front: Optional[Tuple[np.ndarray, List[Dict], np.ndarray]] = None,
    objective_fns: Optional[List[Callable]] = None,
    objective_names: Optional[List[str]] = None,
    baseline_design: Optional[np.ndarray] = None,
    constraint_set=None,
) -> str:
    """
    Generate a comprehensive design report.
    
    Includes:
    - Single best design explanation
    - Pareto front explanations (if provided)
    - Baseline comparisons
    - Sensitivity analysis
    """
    lines = ["AERIS DESIGN REPORT", "=" * 50, ""]
    
    # Best design
    best_explanation = explain_optimization(
        evaluator, optimal_design, objective_fns[0] if objective_fns else None,
        baseline_design=baseline_design, constraint_set=constraint_set
    )
    lines.append(best_explanation.to_text())
    
    # Pareto front
    if pareto_front is not None:
        lines.append("\n\nPARETO FRONT ANALYSIS")
        lines.append("=" * 50)
        
        pareto_x, pareto_metrics, pareto_scores = pareto_front
        pareto_explanations = explain_pareto_front(
            evaluator, pareto_x, pareto_metrics, pareto_scores,
            objective_names or ["obj1", "obj2"],
            objective_fns or [objective_fns[0]] if objective_fns else [],
            baseline_design=baseline_design
        )
        
        for exp in pareto_explanations:
            lines.append("")
            lines.append(exp.to_text())
    
    return "\n".join(lines)