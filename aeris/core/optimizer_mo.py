"""
aeris.core.optimizer_mo
========================
Multi-objective optimization using NSGA-II (Non-dominated Sorting Genetic Algorithm II).

This module provides multi-objective optimization capabilities while maintaining
full compatibility with the existing single-objective differential evolution optimizer.
No external dependencies beyond numpy and scipy.

Reference: Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
IEEE Transactions on Evolutionary Computation, 2002.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional, Tuple, Union

from aeris.core.evaluator_base import Evaluator


@dataclass
class MOOptimizationResult:
    """Result of multi-objective optimization."""
    pareto_x: np.ndarray                    # shape (n_solutions, n_vars)
    pareto_metrics: List[Dict[str, Any]]    # metrics for each Pareto solution
    pareto_scores: np.ndarray               # shape (n_solutions, n_objectives)
    history: List[Dict[str, Any]]           # all evaluations
    generation_log: List[Dict[str, Any]]    # per-generation Pareto fronts
    success: bool
    n_evaluations: int
    
    def summary(self, n_show: int = 5) -> str:
        lines = [f"Pareto front size: {len(self.pareto_x)}", 
                 f"Total evaluations: {self.n_evaluations}"]
        lines.append(f"\nTop {min(n_show, len(self.pareto_x))} Pareto solutions:")
        
        # Sort by first objective (assuming maximization)
        sorted_idx = np.argsort(-self.pareto_scores[:, 0])
        for i, idx in enumerate(sorted_idx[:n_show]):
            lines.append(f"\n  Rank {i+1}:")
            for k, v in zip(self.pareto_metrics[idx].get("_var_names", []), self.pareto_x[idx]):
                lines.append(f"    {k:22s} = {v:.5f}")
            lines.append(f"    Objectives: {self.pareto_scores[idx]}")
            for k, v in self.pareto_metrics[idx].items():
                if not k.startswith("_") and k not in ("constraints", "_var_names", "_validity_conditions"):
                    lines.append(f"    {k:22s} = {v}")
        return "\n".join(lines)


class NSGA2Optimizer:
    """
    NSGA-II Multi-Objective Optimizer.
    
    Features:
    - Fast non-dominated sorting
    - Crowding distance for diversity preservation
    - Tournament selection with crowding comparison
    - Simulated binary crossover (SBX)
    - Polynomial mutation
    - Elitism (combining parent and offspring populations)
    """
    
    def __init__(
        self,
        evaluator: Evaluator,
        objective_fns: List[Callable[[Dict], float]],
        maximize: List[bool] = None,
        popsize: int = 100,
        maxiter: int = 200,
        seed: int = None,
        crossover_prob: float = 0.9,
        crossover_eta: float = 15.0,
        mutation_prob: float = None,
        mutation_eta: float = 20.0,
        penalty: float = 1e6,
        on_generation: Callable = None,
    ):
        """
        Initialize NSGA-II optimizer.
        
        Args:
            evaluator: Evaluator instance
            objective_fns: List of objective functions (metrics -> float)
            maximize: List of booleans, True for maximization per objective
            popsize: Population size (should be even for NSGA-II)
            maxiter: Number of generations
            seed: Random seed
            crossover_prob: SBX crossover probability
            crossover_eta: SBX distribution index
            mutation_prob: Mutation probability (default: 1/n_vars)
            mutation_eta: Polynomial mutation distribution index
            penalty: Penalty for invalid designs
            on_generation: Callback(gen, pareto_x, pareto_scores, pareto_metrics)
        """
        self.evaluator = evaluator
        self.objective_fns = objective_fns
        self.n_objectives = len(objective_fns)
        
        if maximize is None:
            self.maximize = [True] * self.n_objectives
        else:
            self.maximize = maximize
            
        self.popsize = popsize if popsize % 2 == 0 else popsize + 1
        self.maxiter = maxiter
        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.mutation_prob = mutation_prob or (1.0 / evaluator.n_vars())
        self.mutation_eta = mutation_eta
        self.penalty = penalty
        self.on_generation = on_generation
        
        if seed is not None:
            np.random.seed(seed)
            
        self.bounds = np.array(evaluator.bounds)
        self.var_names = evaluator.var_names
        self.n_vars = evaluator.n_vars()
        
        # History tracking
        self.history = []
        self.generation_log = []
        self.n_evaluations = 0
    
    def _evaluate_individual(self, x: np.ndarray) -> Tuple[np.ndarray, Dict, bool]:
        """Evaluate a single individual, return (scores, metrics, valid)."""
        metrics = self.evaluator.evaluate(x)
        self.n_evaluations += 1
        valid = metrics.get("_valid", True)
        
        scores = np.zeros(self.n_objectives)
        for i, fn in enumerate(self.objective_fns):
            try:
                scores[i] = fn(metrics) if valid else -np.inf
            except Exception:
                scores[i] = -np.inf
        
        # Convert to minimization for internal use
        for i in range(self.n_objectives):
            if self.maximize[i]:
                scores[i] = -scores[i]
        
        self.history.append({
            "x": list(map(float, x)),
            "metrics": metrics,
            "scores": scores.copy(),
            "valid": valid,
        })
        
        return scores, metrics, valid
    
    def _dominates(self, scores_a: np.ndarray, scores_b: np.ndarray) -> bool:
        """Check if a dominates b (both in minimization space)."""
        # a dominates b if a is better or equal in all objectives and strictly better in at least one
        return np.all(scores_a <= scores_b) and np.any(scores_a < scores_b)
    
    def _fast_non_dominated_sort(self, scores: np.ndarray) -> List[List[int]]:
        """Fast non-dominated sorting. Returns list of fronts (each front is list of indices)."""
        n = len(scores)
        domination_count = np.zeros(n, dtype=int)
        dominated_sets = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(scores[i], scores[j]):
                    dominated_sets[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(scores[j], scores[i]):
                    dominated_sets[j].append(i)
                    domination_count[i] += 1
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_sets[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            front_idx += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def _crowding_distance(self, scores: np.ndarray, front: List[int]) -> np.ndarray:
        """Calculate crowding distance for individuals in a front."""
        if len(front) <= 2:
            return np.full(len(front), np.inf)
        
        distances = np.zeros(len(front))
        n_obj = scores.shape[1]
        
        for m in range(n_obj):
            # Sort front by objective m
            sorted_idx = np.argsort(scores[front, m])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            
            f_min = scores[front[sorted_idx[0]], m]
            f_max = scores[front[sorted_idx[-1]], m]
            
            if f_max == f_min:
                continue
                
            for k in range(1, len(front) - 1):
                idx = sorted_idx[k]
                prev_idx = sorted_idx[k - 1]
                next_idx = sorted_idx[k + 1]
                distances[idx] += (scores[front[next_idx], m] - scores[front[prev_idx], m]) / (f_max - f_min)
        
        return distances
    
    def _tournament_selection(
        self, 
        population: np.ndarray, 
        scores: np.ndarray, 
        fronts: List[List[int]],
        crowding: Dict[int, float],
        tournament_size: int = 2
    ) -> int:
        """Tournament selection based on rank and crowding distance."""
        candidates = np.random.choice(len(population), tournament_size, replace=False)
        
        best = candidates[0]
        for cand in candidates[1:]:
            # Compare ranks (front index)
            rank_best = next(i for i, f in enumerate(fronts) if best in f)
            rank_cand = next(i for i, f in enumerate(fronts) if cand in f)
            
            if rank_cand < rank_best:
                best = cand
            elif rank_cand == rank_best:
                # Same rank, compare crowding distance (higher is better)
                if crowding[cand] > crowding[best]:
                    best = cand
        
        return best
    
    def _sbx_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX)."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(self.n_vars):
            if np.random.random() > self.crossover_prob:
                continue
                
            x1, x2 = parent1[i], parent2[i]
            if abs(x1 - x2) < 1e-12:
                continue
                
            # Ensure x1 < x2
            if x1 > x2:
                x1, x2 = x2, x1
                
            xl, xu = self.bounds[i]
            beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
            alpha = 2.0 - beta ** -(self.crossover_eta + 1)
            rand = np.random.random()
            
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (self.crossover_eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.crossover_eta + 1))
            
            c1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
            c2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))
            
            # Clip to bounds
            c1 = np.clip(c1, xl, xu)
            c2 = np.clip(c2, xl, xu)
            
            child1[i], child2[i] = c1, c2
        
        return child1, child2
    
    def _polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation."""
        mutant = individual.copy()
        
        for i in range(self.n_vars):
            if np.random.random() > self.mutation_prob:
                continue
                
            x = mutant[i]
            xl, xu = self.bounds[i]
            
            delta1 = (x - xl) / (xu - xl)
            delta2 = (xu - x) / (xu - xl)
            
            rand = np.random.random()
            mut_pow = 1.0 / (self.mutation_eta + 1)
            
            if rand <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.mutation_eta + 1))
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (self.mutation_eta + 1))
                delta_q = 1.0 - val ** mut_pow
            
            x_new = x + delta_q * (xu - xl)
            mutant[i] = np.clip(x_new, xl, xu)
        
        return mutant
    
    def _initialize_population(self) -> np.ndarray:
        """Initialize random population within bounds."""
        pop = np.zeros((self.popsize, self.n_vars))
        for i in range(self.n_vars):
            xl, xu = self.bounds[i]
            pop[:, i] = np.random.uniform(xl, xu, self.popsize)
        return pop
    
    def optimize(self) -> MOOptimizationResult:
        """Run NSGA-II optimization."""
        # Initialize population
        population = self._initialize_population()
        
        # Evaluate initial population
        scores = np.zeros((self.popsize, self.n_objectives))
        metrics_list = []
        valid_list = []
        
        for i in range(self.popsize):
            s, m, v = self._evaluate_individual(population[i])
            scores[i] = s
            metrics_list.append(m)
            valid_list.append(v)
        
        # Main loop
        for gen in range(self.maxiter):
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(scores)
            
            # Crowding distance
            crowding = {}
            for front_idx, front in enumerate(fronts):
                if front:
                    cd = self._crowding_distance(scores, front)
                    for idx, dist in zip(front, cd):
                        crowding[idx] = dist
            
            # Log generation
            pareto_front = fronts[0] if fronts else []
            self.generation_log.append({
                "generation": gen,
                "pareto_x": population[pareto_front].copy() if pareto_front else np.array([]),
                "pareto_scores": scores[pareto_front].copy() if pareto_front else np.array([]),
                "pareto_metrics": [metrics_list[i] for i in pareto_front],
                "fronts_count": len(fronts),
                "pop_valid": sum(valid_list),
            })
            
            if self.on_generation is not None:
                pareto_x = population[pareto_front] if pareto_front else np.array([])
                pareto_s = scores[pareto_front] if pareto_front else np.array([])
                pareto_m = [metrics_list[i] for i in pareto_front]
                self.on_generation(gen, pareto_x, pareto_s, pareto_m)
            
            # Create offspring
            offspring = np.zeros((self.popsize, self.n_vars))
            offspring_scores = np.zeros((self.popsize, self.n_objectives))
            offspring_metrics = []
            offspring_valid = []
            
            for i in range(0, self.popsize, 2):
                # Selection
                p1_idx = self._tournament_selection(population, scores, fronts, crowding)
                p2_idx = self._tournament_selection(population, scores, fronts, crowding)
                
                # Crossover
                child1, child2 = self._sbx_crossover(population[p1_idx], population[p2_idx])
                
                # Mutation
                child1 = self._polynomial_mutation(child1)
                child2 = self._polynomial_mutation(child2)
                
                offspring[i] = child1
                if i + 1 < self.popsize:
                    offspring[i + 1] = child2
            
            # Evaluate offspring
            for i in range(self.popsize):
                s, m, v = self._evaluate_individual(offspring[i])
                offspring_scores[i] = s
                offspring_metrics.append(m)
                offspring_valid.append(v)
            
            # Combine parent and offspring
            combined_pop = np.vstack([population, offspring])
            combined_scores = np.vstack([scores, offspring_scores])
            combined_metrics = metrics_list + offspring_metrics
            combined_valid = valid_list + offspring_valid
            
            # Non-dominated sorting on combined population
            combined_fronts = self._fast_non_dominated_sort(combined_scores)
            
            # Crowding distance for combined
            combined_crowding = {}
            for front_idx, front in enumerate(combined_fronts):
                if front:
                    cd = self._crowding_distance(combined_scores, front)
                    for idx, dist in zip(front, cd):
                        combined_crowding[idx] = dist
            
            # Select next generation
            new_pop = []
            new_scores = []
            new_metrics = []
            new_valid = []
            
            for front in combined_fronts:
                if len(new_pop) + len(front) <= self.popsize:
                    # Take entire front
                    for idx in front:
                        new_pop.append(combined_pop[idx])
                        new_scores.append(combined_scores[idx])
                        new_metrics.append(combined_metrics[idx])
                        new_valid.append(combined_valid[idx])
                else:
                    # Sort front by crowding distance (descending) and take needed
                    front_sorted = sorted(front, key=lambda idx: -combined_crowding[idx])
                    needed = self.popsize - len(new_pop)
                    for idx in front_sorted[:needed]:
                        new_pop.append(combined_pop[idx])
                        new_scores.append(combined_scores[idx])
                        new_metrics.append(combined_metrics[idx])
                        new_valid.append(combined_valid[idx])
                    break
            
            population = np.array(new_pop)
            scores = np.array(new_scores)
            metrics_list = new_metrics
            valid_list = new_valid
        
        # Final Pareto front
        final_fronts = self._fast_non_dominated_sort(scores)
        pareto_indices = final_fronts[0] if final_fronts else []
        
        pareto_x = population[pareto_indices]
        pareto_scores = scores[pareto_indices]
        pareto_metrics = [metrics_list[i] for i in pareto_indices]
        
        # Convert scores back to original sense (maximization)
        final_scores = pareto_scores.copy()
        for i in range(self.n_objectives):
            if self.maximize[i]:
                final_scores[:, i] = -final_scores[:, i]
        
        # Add var_names to metrics
        for m in pareto_metrics:
            m["_var_names"] = self.var_names
        
        return MOOptimizationResult(
            pareto_x=pareto_x,
            pareto_metrics=pareto_metrics,
            pareto_scores=final_scores,
            history=self.history,
            generation_log=self.generation_log,
            success=len(pareto_indices) > 0,
            n_evaluations=self.n_evaluations,
        )


def optimize(
    evaluator: Evaluator,
    objective_fn: Union[Callable, List[Callable]],
    maximize: Union[bool, List[bool]] = True,
    popsize: int = 20,
    maxiter: int = 100,
    seed: int = None,
    workers: int = 1,
    penalty: float = 1e6,
    on_generation: Callable = None,
    multi_objective: bool = False,
) -> Union["OptimizationResult", MOOptimizationResult]:
    """
    Unified optimization interface supporting both single and multi-objective.
    
    Args:
        evaluator: Evaluator instance
        objective_fn: Single function or list of functions (metrics -> float)
        maximize: Single bool or list of bools per objective
        popsize: Population size
        maxiter: Maximum iterations/generations
        seed: Random seed
        workers: Number of workers (only for single-objective DE)
        penalty: Penalty for invalid designs
        on_generation: Callback function
        multi_objective: If True, use NSGA-II multi-objective optimization
        
    Returns:
        SingleOptimizationResult or MOOptimizationResult
    """
    if multi_objective:
        if not isinstance(objective_fn, list):
            objective_fn = [objective_fn]
        if not isinstance(maximize, list):
            maximize = [maximize] * len(objective_fn)
        
        optimizer = NSGA2Optimizer(
            evaluator=evaluator,
            objective_fns=objective_fn,
            maximize=maximize,
            popsize=popsize,
            maxiter=maxiter,
            seed=seed,
            penalty=penalty,
            on_generation=on_generation,
        )
        return optimizer.optimize()
    else:
        # Inline single-objective DE to avoid circular import
        from scipy.optimize import differential_evolution
        
        class OptimizationResult:
            def __init__(self, x, metrics, score, history, generation_log, success):
                self.x = x
                self.metrics = metrics
                self.score = score
                self.history = history
                self.generation_log = generation_log
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

# Need to define the base class for type annotation
class OptimizationResult:
    pass