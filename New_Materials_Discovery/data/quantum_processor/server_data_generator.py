# data_generator_quantum_server.py
# Author: Cette
# Version: 3.1.0 - Enterprise Cloud Edition

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm, lognorm
import itertools
from tqdm import tqdm
import scipy.constants as const
import networkx as nx
from math import ceil


class QubitsConfiguration:
    """Configuration constants for quantum server"""
    CONTROL_LINES = 2  # DC bias lines per qubit
    RF_LINES = 1  # RF control line per qubit
    READOUT_LINES = 1  # Readout line per qubit
    BASE_QUBIT_SIZE = 50  # μm²
    BASE_DISSIPATION = 1e-15  # W per qubit
    CONTROL_DISSIPATION = 1e-14  # W per line
    RACK_UNIT_COST = 50000  # USD per rack unit
    COOLING_COST_PER_WATT = 1e6  # USD per Watt at 20mK


class QuantumServerGenerator:
    """Generate quantum server configurations for cloud deployment"""

    def __init__(self):
        self.host_materials = {
            'semiconductors': ['Si', 'Ge', 'GaAs', 'InP', 'SiC'],
            'diamonds': ['C', 'SiV', 'NV', 'GeV'],
            'quantum_dots': ['InAs', 'InGaAs', 'GaN', 'CdSe'],
            'superconductors': ['Al', 'Nb', 'NbN', 'NbTiN']
        }

        self.server_configs = {
            'rack_units': [1, 2, 4, 8],
            'qubits_per_unit': [100, 500, 1000, 5000],
            'interconnect_topologies': ['line', 'grid', 'hexagonal', 'fully_connected'],
            'cooling_systems': ['dilution_refrigerator', 'adiabatic_demagnetization'],
            'control_electronics': ['FPGA', 'ASIC', 'hybrid']
        }

        self.infrastructure = {
            'power_budget': (1, 50),  # kW per rack
            'cooling_capacity': (0.1, 2),  # W at 20mK
            'control_channels': (100, 10000),
            'readout_channels': (100, 5000),
            'network_bandwidth': (10, 100)  # Gbps
        }

        self.quantum_constants = {
            'h_planck': const.h,
            'hbar': const.hbar,
            'k_boltzmann': const.k,
            'e_charge': const.e,
            'base_temperature': 0.020,
            'magnetic_field': 1.0
        }

        self.default_material_params = {
            'Si': {
                'lattice_constant': 5.431,
                'spin_orbit_coupling': 44e-6,
                'g_factor': 2.0,
                'hyperfine_coupling': 117e6,
                'debye_temperature': 645,
                'fab_compatibility': 0.95,
                'control_complexity': 0.8,
                'yield_at_scale': 0.92,
                'cost_per_qubit': 100
            }
        }

        self.material_parameters = self._initialize_material_parameters()

    def _initialize_material_parameters(self) -> Dict:
        """Initialize material parameters with defaults for unlisted materials"""
        material_params = self.default_material_params.copy()
        all_materials = [m for sublist in self.host_materials.values() for m in sublist]

        for material in all_materials:
            if material not in material_params:
                material_params[material] = self._generate_material_params()

        return material_params

    def _generate_material_params(self) -> Dict:
        """Generate parameters for new materials"""
        return {
            'lattice_constant': np.random.uniform(3.5, 6.5),
            'spin_orbit_coupling': np.random.uniform(1e-6, 500e-6),
            'g_factor': np.random.uniform(0.5, 2.5),
            'hyperfine_coupling': np.random.uniform(1e6, 200e6),
            'debye_temperature': np.random.uniform(300, 1000),
            'fab_compatibility': np.random.uniform(0.6, 0.9),
            'control_complexity': np.random.uniform(0.7, 0.9),
            'yield_at_scale': np.random.uniform(0.8, 0.95),
            'cost_per_qubit': np.random.uniform(50, 500)
        }

    def generate_server_architecture(
            self,
            n_qubits: int,
            topology: str,
            material: str
    ) -> Dict:
        """Generate complete server architecture specification"""
        params = self.material_parameters[material]

        # Calculate layout
        area_per_qubit = self._calculate_qubit_area(material)
        total_area = area_per_qubit * n_qubits * 1.5

        control = self._calculate_control_requirements(n_qubits, topology)
        cooling = self._calculate_cooling_requirements(n_qubits, material)

        # Determine rack configuration
        viable_racks = [ru for ru in self.server_configs['rack_units']
                        if ru * 1000 >= total_area]
        rack_units = min(viable_racks) if viable_racks else max(self.server_configs['rack_units'])

        return {
            'physical_layout': {
                'area_per_qubit': area_per_qubit,
                'total_area': total_area,
                'rack_units': rack_units,
                'density': n_qubits / total_area
            },
            'control_system': control,
            'cooling_system': {
                'power_at_20mK': cooling,
                'required_capacity': cooling * 1.5
            },
            'expected_yield': params['yield_at_scale'],
            'cost_estimate': self._calculate_cost(n_qubits, material, rack_units, cooling)
        }

    def _calculate_qubit_area(self, material: str) -> float:
        """Calculate area needed per qubit including control lines"""
        params = self.material_parameters[material]
        return QubitsConfiguration.BASE_QUBIT_SIZE * (1 / params['fab_compatibility']) * \
            (1 + params['control_complexity'])

    def _calculate_control_requirements(self, n_qubits: int, topology: str) -> Dict:
        """Calculate control system specifications"""
        topology_overhead = {
            'fully_connected': 0.5,
            'grid': 0.3,
            'hexagonal': 0.25,
            'line': 0.2
        }
        overhead = topology_overhead.get(topology, 0.2)

        dc_total = int(n_qubits * QubitsConfiguration.CONTROL_LINES * (1 + overhead))
        rf_total = int(n_qubits * QubitsConfiguration.RF_LINES * (1 + overhead))
        readout_total = int(n_qubits * QubitsConfiguration.READOUT_LINES)

        total_channels = dc_total + rf_total + readout_total

        return {
            'dc_channels': dc_total,
            'rf_channels': rf_total,
            'readout_channels': readout_total,
            'total_channels': total_channels,
            'required_fpgas': ceil(total_channels / 100),
            'control_latency': self._estimate_control_latency(topology, n_qubits)
        }

    def _calculate_cooling_requirements(self, n_qubits: int, material: str) -> float:
        """Calculate required cooling power at 20mK"""
        params = self.material_parameters[material]
        material_factor = 1 + (1 - params['fab_compatibility'])
        lines_per_qubit = (QubitsConfiguration.CONTROL_LINES +
                           QubitsConfiguration.RF_LINES +
                           QubitsConfiguration.READOUT_LINES)

        return n_qubits * (
                QubitsConfiguration.BASE_DISSIPATION +
                QubitsConfiguration.CONTROL_DISSIPATION * lines_per_qubit
        ) * material_factor

    def _calculate_cost(
            self,
            n_qubits: int,
            material: str,
            rack_units: int,
            cooling_power: float
    ) -> Dict[str, float]:
        """Calculate comprehensive cost breakdown"""
        params = self.material_parameters[material]

        fab_cost = n_qubits * params['cost_per_qubit']
        control_cost = self._calculate_control_cost(n_qubits)
        cooling_cost = cooling_power * QubitsConfiguration.COOLING_COST_PER_WATT
        infrastructure_cost = rack_units * QubitsConfiguration.RACK_UNIT_COST

        total_cost = sum([fab_cost, control_cost, cooling_cost, infrastructure_cost])

        return {
            'fabrication': fab_cost,
            'control_electronics': control_cost,
            'cooling_system': cooling_cost,
            'infrastructure': infrastructure_cost,
            'total': total_cost,
            'per_qubit': total_cost / n_qubits
        }

    def _calculate_coherence_time(self, params: Dict, temperature: float) -> float:
        """Calculate qubit coherence time based on material parameters and temperature"""
        # Cap temperature ratio to avoid overflow
        temp_ratio = min(params['debye_temperature'] / temperature, 100)

        # Base coherence affected by spin-orbit coupling and temperature
        base_coherence = 1 / (params['spin_orbit_coupling'] * temperature)

        # Scale by material quality factors
        material_factor = params['fab_compatibility'] * params['yield_at_scale']

        # Temperature dependence with capped exponential
        temp_factor = np.exp(temp_ratio)

        # Cap final coherence time
        return min(1e6, base_coherence * material_factor * temp_factor * 1e-6)

    def _calculate_gate_time(self, params: Dict) -> float:
        """Calculate single-qubit gate operation time"""
        # Base gate time proportional to hyperfine coupling with minimum value
        base_gate_time = max(1e-9, 1 / params['hyperfine_coupling'])

        # Scale by control complexity
        control_factor = 1 + params['control_complexity']

        return base_gate_time * control_factor

    def _calculate_fidelity(self, params: Dict, temperature: float) -> float:
        """Calculate single-qubit gate fidelity"""
        # Base fidelity dependent on fabrication quality
        base_fidelity = params['fab_compatibility']

        # Temperature effects with capped ratio
        temp_factor = 1 - min(1, (temperature / params['debye_temperature']))

        # Control precision effects
        control_factor = 1 - (1 - params['control_complexity']) / 2

        return min(0.9999, base_fidelity * temp_factor * control_factor)

    def _calculate_quantum_volume(self, n_qubits: int, fidelity: float) -> int:
        """Calculate quantum volume metric"""
        # Based on IBM's quantum volume definition with capped values
        depth = min(100, int(np.floor(-np.log2(1 - fidelity))))
        effective_qubits = min(n_qubits, depth)

        return min(2 ** 30, int(2 ** effective_qubits))  # Cap at 2^30 to avoid overflow

    def _calculate_coherence_time(self, params: Dict, temperature: float) -> float:
        """Calculate qubit coherence time based on material parameters and temperature"""
        temp_ratio = min(params['debye_temperature'] / temperature, 100)
        base_coherence = 1 / (params['spin_orbit_coupling'] * temperature)
        material_factor = params['fab_compatibility'] * params['yield_at_scale']
        temp_factor = np.exp(temp_ratio)
        return min(1e6, base_coherence * material_factor * temp_factor * 1e-6)

    def _calculate_gate_time(self, params: Dict) -> float:
        """Calculate single-qubit gate operation time"""
        base_gate_time = max(1e-9, 1 / params['hyperfine_coupling'])
        control_factor = 1 + params['control_complexity']
        return max(1e-9, base_gate_time * control_factor)

    def _calculate_fidelity(self, params: Dict, temperature: float) -> float:
        """Calculate single-qubit gate fidelity"""
        base_fidelity = params['fab_compatibility']
        temp_factor = 1 - min(1, (temperature / params['debye_temperature']))
        control_factor = 1 - (1 - params['control_complexity']) / 2
        return min(0.9999, base_fidelity * temp_factor * control_factor)

    def _calculate_quantum_volume(self, n_qubits: int, fidelity: float) -> int:
        """Calculate quantum volume metric"""
        depth = min(100, int(np.floor(-np.log2(1 - fidelity))))
        effective_qubits = min(n_qubits, depth)
        return min(2 ** 30, int(2 ** effective_qubits))

    def generate_quantum_properties(self, material: str, n_qubits: int) -> Dict:
        """Generate quantum properties scaled to server level"""
        params = self.material_parameters[material]
        temperature = self.quantum_constants['base_temperature']

        coherence_time = self._calculate_coherence_time(params, temperature)
        gate_time = self._calculate_gate_time(params)
        fidelity = self._calculate_fidelity(params, temperature)
        scale_factor = 1 / np.log10(n_qubits + 10)

        # Ensure circuit depth doesn't overflow
        max_depth = min(int(1e6), int(coherence_time / max(1e-9, gate_time) * scale_factor))

        return {
            'coherence_time': coherence_time,
            'gate_time': gate_time,
            'single_qubit_fidelity': fidelity,
            'two_qubit_fidelity': fidelity ** 2,
            'quantum_volume': self._calculate_quantum_volume(n_qubits, fidelity),
            'max_circuit_depth': max_depth
        }

    def generate_server_dataset(self, n_servers: int = 1000, batch_size: int = 50) -> pd.DataFrame:
        """Generate server configurations in batches to manage memory"""
        data = []
        n_batches = (n_servers + batch_size - 1) // batch_size

        for batch in tqdm(range(n_batches), desc="Generating server configurations"):
            batch_size = min(batch_size, n_servers - len(data))
            for _ in range(batch_size):
                n_qubits = np.random.choice(self.server_configs['qubits_per_unit']) * \
                           np.random.choice(self.server_configs['rack_units'])
                topology = np.random.choice(self.server_configs['interconnect_topologies'])
                material = np.random.choice(list(self.material_parameters.keys()))

                server_arch = self.generate_server_architecture(n_qubits, topology, material)
                quantum_props = self.generate_quantum_properties(material, n_qubits)
                network = self.generate_network_topology(n_qubits, topology)

                data.append({
                    'n_qubits': n_qubits,
                    'topology': topology,
                    'material': material,
                    **server_arch,
                    **quantum_props,
                    'network_metrics': self._calculate_network_metrics(network)
                })

            # Free memory by creating DataFrame and clearing list periodically
            if len(data) >= batch_size * 10:
                temp_df = pd.DataFrame(data)
                data = []
                yield temp_df

        if data:
            yield pd.DataFrame(data)

    def generate_quantum_properties(self, material: str, n_qubits: int) -> Dict:
        """Generate quantum properties scaled to server level"""
        params = self.material_parameters[material]
        temperature = self.quantum_constants['base_temperature']

        # Base properties
        coherence_time = self._calculate_coherence_time(params, temperature)
        gate_time = self._calculate_gate_time(params)
        fidelity = self._calculate_fidelity(params, temperature)

        # Scale to server level
        scale_factor = 1 / np.log10(n_qubits + 10)

        return {
            'coherence_time': coherence_time * scale_factor,
            'gate_time': gate_time,
            'single_qubit_fidelity': fidelity,
            'two_qubit_fidelity': fidelity ** 2,
            'quantum_volume': self._calculate_quantum_volume(n_qubits, fidelity),
            'max_circuit_depth': int(coherence_time / gate_time * scale_factor)
        }

    def generate_network_topology(self, n_qubits: int, topology: str) -> nx.Graph:
        """Generate simplified network topology"""
        # Limit maximum number of qubits to prevent hanging
        n_qubits = min(n_qubits, 1000)

        if topology == 'line':
            G = nx.path_graph(n_qubits)
        elif topology == 'grid':
            side = int(np.ceil(np.sqrt(n_qubits)))
            G = nx.grid_2d_graph(min(side, 32), min(side, 32))
        elif topology == 'hexagonal':
            side = int(np.sqrt(n_qubits / 2))
            G = nx.hexagonal_lattice_graph(min(side, 16), min(side, 16))
        else:  # fully_connected
            G = nx.complete_graph(min(n_qubits, 100))

        # Simplified edge properties
        for u, v in G.edges():
            G[u][v]['coupling'] = 0.5
            G[u][v]['fidelity'] = 0.95

        return G

    def _calculate_network_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """Calculate basic network metrics with timeout"""
        try:
            return {
                'connectivity': float(nx.average_node_connectivity(G)),
                'diameter': float(nx.diameter(G)),
                'clustering': float(nx.average_clustering(G)),
                'efficiency': float(nx.global_efficiency(G))
            }
        except:
            return {
                'connectivity': 1.0,
                'diameter': 1.0,
                'clustering': 1.0,
                'efficiency': 1.0
            }

    def _estimate_control_latency(self, topology: str, n_qubits: int) -> float:
        """Estimate control system latency"""
        base_latency = 100e-9  # 100 ns base
        topology_factor = {
            'line': 1.0,
            'grid': 1.2,
            'hexagonal': 1.3,
            'fully_connected': 1.5
        }
        return base_latency * topology_factor.get(topology, 1.0) * np.log10(n_qubits + 1)

    def _calculate_control_cost(self, n_qubits: int) -> float:
        """Calculate control electronics cost"""
        return n_qubits * 500  # $500 per qubit for control electronics

    def generate_network_topology(self, n_qubits: int, topology: str) -> nx.Graph:
        """Generate simplified network topology"""
        n_qubits = min(n_qubits, 100)  # Limit size

        if topology == 'line':
            G = nx.path_graph(n_qubits)
        elif topology == 'grid':
            side = min(int(np.ceil(np.sqrt(n_qubits))), 10)
            G = nx.grid_2d_graph(side, side)
        elif topology == 'hexagonal':
            side = min(int(np.sqrt(n_qubits / 2)), 8)
            G = nx.hexagonal_lattice_graph(side, side)
        else:
            G = nx.complete_graph(min(n_qubits, 50))

        for u, v in G.edges():
            G[u][v].update({'coupling': 0.5, 'fidelity': 0.95})
        return G

    def _calculate_network_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """Calculate basic network metrics"""
        try:
            return {
                'connectivity': float(nx.average_node_connectivity(G)),
                'diameter': float(nx.diameter(G)),
                'clustering': float(nx.average_clustering(G)),
                'efficiency': float(nx.global_efficiency(G))
            }
        except:
            return {
                'connectivity': 1.0,
                'diameter': 1.0,
                'clustering': 1.0,
                'efficiency': 1.0
            }

    def generate_server_dataset(self, n_servers: int = 20) -> pd.DataFrame:
        """Generate server configurations with smaller batches"""
        data = []

        for _ in tqdm(range(n_servers), desc="Generating server configurations"):
            try:
                n_qubits = np.random.choice(self.server_configs['qubits_per_unit']) * \
                           np.random.choice([1, 2])  # Reduced rack units
                topology = np.random.choice(self.server_configs['interconnect_topologies'])
                material = np.random.choice(list(self.material_parameters.keys()))

                server_arch = self.generate_server_architecture(n_qubits, topology, material)
                quantum_props = self.generate_quantum_properties(material, n_qubits)
                network = self.generate_network_topology(n_qubits, topology)

                data.append({
                    'n_qubits': n_qubits,
                    'topology': topology,
                    'material': material,
                    **server_arch,
                    **quantum_props,
                    'network_metrics': self._calculate_network_metrics(network)
                })
            except Exception as e:
                print(f"Error generating server {_}: {str(e)}")
                continue

        return pd.DataFrame(data)


if __name__ == "__main__":
    generator = QuantumServerGenerator()
    print("Generating quantum server configurations...")
    server_df = generator.generate_server_dataset(n_servers=20)

    # Flatten cost_estimate dictionary before saving
    cost_cols = []
    for idx, row in server_df.iterrows():
        for cost_key, cost_val in row['cost_estimate'].items():
            server_df.at[idx, f'cost_{cost_key}'] = cost_val

    server_df.to_csv("quantum_server_configs.csv", index=False)

    print("\nServer Configuration Statistics:")
    print(f"Average qubits per server: {server_df['n_qubits'].mean():.0f}")
    print(f"Average cost per qubit: ${server_df['cost_per_qubit'].mean():.2f}")
    print(f"Most common topology: {server_df['topology'].mode().iloc[0]}")
    print(f"Most common material: {server_df['material'].mode().iloc[0]}")