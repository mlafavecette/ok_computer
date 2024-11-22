import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import yaml
import json
from scipy.optimize import minimize, LinearConstraint
from datetime import datetime
import os

# Local imports assuming project structure
from .constraints import ProcessConstraints
from .utils import EmissionCalculator
import sys

sys.path.append("../../data/biofuel_data")
from data_generator import EmissionFactors


class OptimizationObjective(Enum):
    MINIMIZE_EMISSIONS = "minimize_emissions"
    MAXIMIZE_YIELD = "maximize_yield"
    OPTIMIZE_COST = "optimize_cost"
    BALANCE_ALL = "balance_all"


@dataclass
class OptimizationResults:
    """Container for optimization results"""
    optimized_parameters: Dict[str, float]
    emission_reduction: float
    yield_improvement: float
    cost_savings: float
    roi_months: float
    implementation_steps: List[str]
    monitoring_points: List[str]


class EmissionOptimizer:
    """
    Industrial process optimizer for plastic-to-fuel conversion.
    Focuses on emission reduction while balancing yield and cost.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.emission_factors = EmissionFactors()
        self.calculator = EmissionCalculator()

        # Load configuration
        config_path = config_path or "../config/emission_factors.yaml"
        self.config = self._load_config(config_path)

        # Initialize constraints
        self.constraints = ProcessConstraints()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._get_default_config()

    def optimize_process(self,
                         current_params: Dict[str, float],
                         objective: OptimizationObjective,
                         constraints: Optional[Dict] = None) -> OptimizationResults:
        """
        Optimize process parameters for given objective.

        Args:
            current_params: Current process parameters including:
                - reactor_temperature (°C)
                - pressure (bar)
                - microwave_power (kW)
                - catalyst_loading (kg/L)
                - steam_injection_rate (kg/hour)
                - energy_mix (Dict[str, float])
                - residence_time (minutes)
            objective: Optimization objective from OptimizationObjective enum
            constraints: Optional additional constraints

        Returns:
            OptimizationResults with optimized parameters and improvements
        """
        self.logger.info(f"Starting process optimization for {objective.value}")

        # Update constraints if provided
        if constraints:
            self.constraints.update(constraints)

        # Define optimization bounds
        bounds = self._get_parameter_bounds()

        # Choose objective function based on optimization goal
        objective_function = self._get_objective_function(objective)

        # Initialize optimization
        x0 = self._params_to_array(current_params)
        constraints = self._get_optimization_constraints()

        # Run optimization
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            self.logger.warning("Optimization did not converge")

        # Convert results back to parameter space
        optimized_params = self._array_to_params(result.x)

        # Calculate improvements
        improvements = self._calculate_improvements(
            current_params, optimized_params)

        return OptimizationResults(
            optimized_parameters=optimized_params,
            emission_reduction=improvements['emission_reduction'],
            yield_improvement=improvements['yield_improvement'],
            cost_savings=improvements['cost_savings'],
            roi_months=improvements['roi_months'],
            implementation_steps=self._generate_implementation_steps(
                current_params, optimized_params),
            monitoring_points=self._generate_monitoring_points(optimized_params)
        )

    def _get_objective_function(self, objective: OptimizationObjective):
        """Return appropriate objective function based on optimization goal"""

        def emissions_objective(x: np.ndarray) -> float:
            params = self._array_to_params(x)
            return self.calculator.calculate_total_emissions(params)

        def yield_objective(x: np.ndarray) -> float:
            params = self._array_to_params(x)
            return -self.calculator.calculate_yield(params)  # Negative for maximization

        def cost_objective(x: np.ndarray) -> float:
            params = self._array_to_params(x)
            return self.calculator.calculate_operating_cost(params)

        def balanced_objective(x: np.ndarray) -> float:
            params = self._array_to_params(x)
            emissions = self.calculator.calculate_total_emissions(params)
            cost = self.calculator.calculate_operating_cost(params)
            yield_value = self.calculator.calculate_yield(params)

            # Normalize and combine objectives
            return (0.4 * emissions / 1000 +  # Emissions in tons CO2e
                    0.3 * cost / 1000 +  # Cost in thousands
                    0.3 * (1 - yield_value))  # Yield loss

        objectives = {
            OptimizationObjective.MINIMIZE_EMISSIONS: emissions_objective,
            OptimizationObjective.MAXIMIZE_YIELD: yield_objective,
            OptimizationObjective.OPTIMIZE_COST: cost_objective,
            OptimizationObjective.BALANCE_ALL: balanced_objective
        }

        return objectives[objective]

    def _get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for optimization parameters"""
        return [
            (450, 650),  # reactor_temperature
            (3, 15),  # pressure
            (80, 120),  # microwave_power
            (0.1, 0.5),  # catalyst_loading
            (300, 600),  # steam_injection_rate
            (0.3, 1.0),  # renewable_energy_ratio
            (30, 90)  # residence_time
        ]

    def _get_optimization_constraints(self) -> List[Dict]:
        """Define optimization constraints"""
        return [
            {'type': 'ineq', 'fun': lambda x: self.constraints.max_energy -
                                              (x[2] * x[6] / 60)},  # Energy consumption constraint
            {'type': 'ineq', 'fun': lambda x: self.constraints.min_yield -
                                              self.calculator.calculate_yield(self._array_to_params(x))},
            {'type': 'ineq', 'fun': lambda x: x[5] -
                                              self.constraints.min_renewable_ratio}  # Renewable energy constraint
        ]

    def _calculate_improvements(self,
                                current: Dict[str, float],
                                optimized: Dict[str, float]) -> Dict[str, float]:
        """Calculate improvements from optimization"""
        current_emissions = self.calculator.calculate_total_emissions(current)
        optimized_emissions = self.calculator.calculate_total_emissions(optimized)

        current_yield = self.calculator.calculate_yield(current)
        optimized_yield = self.calculator.calculate_yield(optimized)

        current_cost = self.calculator.calculate_operating_cost(current)
        optimized_cost = self.calculator.calculate_operating_cost(optimized)

        # Calculate ROI
        implementation_cost = self._estimate_implementation_cost(
            current, optimized)
        annual_savings = (current_cost - optimized_cost) * 365
        roi_months = (implementation_cost / annual_savings) * 12

        return {
            'emission_reduction': current_emissions - optimized_emissions,
            'yield_improvement': optimized_yield - current_yield,
            'cost_savings': current_cost - optimized_cost,
            'roi_months': roi_months
        }

    def _generate_implementation_steps(self,
                                       current: Dict[str, float],
                                       optimized: Dict[str, float]) -> List[str]:
        """Generate step-by-step implementation plan"""
        steps = []

        # Temperature adjustment
        if abs(optimized['reactor_temperature'] - current['reactor_temperature']) > 10:
            steps.append(
                f"Gradually adjust reactor temperature to {optimized['reactor_temperature']}°C "
                f"in 10°C increments while monitoring pressure"
            )

        # Pressure changes
        if abs(optimized['pressure'] - current['pressure']) > 1:
            steps.append(
                f"Adjust system pressure to {optimized['pressure']} bar "
                f"while maintaining temperature stability"
            )

        # Catalyst modifications
        if abs(optimized['catalyst_loading'] - current['catalyst_loading']) > 0.05:
            steps.append(
                f"Modify catalyst loading to {optimized['catalyst_loading']} kg/L. "
                f"Consider catalyst regeneration cycle timing"
            )

        # Energy mix optimization
        if optimized['renewable_energy_ratio'] > current.get('renewable_energy_ratio', 0):
            steps.append(
                f"Increase renewable energy usage to {optimized['renewable_energy_ratio'] * 100}% "
                f"of total energy consumption"
            )

        # Process timing
        if abs(optimized['residence_time'] - current['residence_time']) > 5:
            steps.append(
                f"Adjust residence time to {optimized['residence_time']} minutes. "
                f"Update process control parameters accordingly"
            )

        return steps

    def _generate_monitoring_points(self,
                                    optimized: Dict[str, float]) -> List[str]:
        """Generate key monitoring points for optimized process"""
        return [
            f"Monitor reactor temperature at {optimized['reactor_temperature']}°C ±5°C",
            f"Maintain system pressure at {optimized['pressure']} bar ±0.5 bar",
            f"Track catalyst activity daily with regeneration at {optimized['catalyst_loading']}kg/L loading",
            "Monitor product composition hourly via online GC",
            "Track energy consumption and renewable energy ratio hourly",
            f"Ensure residence time stability at {optimized['residence_time']} minutes ±2 minutes",
            "Monitor emissions continuously at stack",
            "Track yield and product quality every 2 hours"
        ]

    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to array for optimization"""
        return np.array([
            params['reactor_temperature'],
            params['pressure'],
            params['microwave_power'],
            params['catalyst_loading'],
            params['steam_injection_rate'],
            params.get('renewable_energy_ratio', 0.3),
            params['residence_time']
        ])

    def _array_to_params(self, x: np.ndarray) -> Dict[str, float]:
        """Convert optimization array back to parameter dictionary"""
        return {
            'reactor_temperature': x[0],
            'pressure': x[1],
            'microwave_power': x[2],
            'catalyst_loading': x[3],
            'steam_injection_rate': x[4],
            'renewable_energy_ratio': x[5],
            'residence_time': x[6]
        }

    def save_optimization_results(self,
                                  results: OptimizationResults,
                                  save_path: str) -> None:
        """Save optimization results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save main results
        results_dict = {
            'timestamp': timestamp,
            'parameters': results.optimized_parameters,
            'improvements': {
                'emission_reduction': float(results.emission_reduction),
                'yield_improvement': float(results.yield_improvement),
                'cost_savings': float(results.cost_savings),
                'roi_months': float(results.roi_months)
            },
            'implementation': {
                'steps': results.implementation_steps,
                'monitoring_points': results.monitoring_points
            }
        }

        with open(save_dir / f'optimization_results_{timestamp}.json', 'w') as f:
            json.dump(results_dict, f, indent=2)


if __name__ == "__main__":
    # Example usage
    optimizer = EmissionOptimizer()

    current_params = {
        'reactor_temperature': 550,
        'pressure': 5,
        'microwave_power': 100,
        'catalyst_loading': 0.3,
        'steam_injection_rate': 450,
        'renewable_energy_ratio': 0.3,
        'residence_time': 45
    }

    results = optimizer.optimize_process(
        current_params,
        OptimizationObjective.BALANCE_ALL
    )

    optimizer.save_optimization_results(
        results,
        "optimization_results"
    )