# data_generator_quantum.py
# Author: Cette
# Version: 2.0.1

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import norm, lognorm
import itertools
from tqdm import tqdm
import scipy.constants as const


class QuantumMaterialGenerator:
    def __init__(self):
        self.host_materials = {
            'semiconductors': ['Si', 'Ge', 'GaAs', 'InP', 'SiC'],
            'diamonds': ['C', 'SiV', 'NV', 'GeV'],
            'quantum_dots': ['InAs', 'InGaAs', 'GaN', 'CdSe'],
            'superconductors': ['Al', 'Nb', 'NbN', 'NbTiN']
        }

        self.defect_types = {
            'vacancies': ['V_Si', 'V_C', 'V_Ga', 'V_As'],
            'interstitials': ['Si_i', 'C_i', 'Ga_i', 'As_i'],
            'substitutionals': ['P_Si', 'B_Si', 'N_C', 'Al_Ga']
        }

        self.property_ranges = {
            'coherence_time': (1e-6, 1e-3),  # s
            'coupling_strength': (1e6, 1e9),  # Hz
            'anharmonicity': (100e6, 500e6),  # Hz
            'transition_frequency': (1e9, 10e9),  # Hz
            'relaxation_time': (1e-6, 1e-3),  # s
            'dephasing_time': (1e-6, 1e-3),  # s
            'gate_fidelity': (0.9, 0.9999),
            'readout_fidelity': (0.9, 0.9999)
        }

        self.quantum_constants = {
            'h_planck': const.h,  # J·s
            'hbar': const.hbar,  # J·s
            'k_boltzmann': const.k,  # J/K
            'e_charge': const.e,  # C
            'base_temperature': 0.020,  # K
            'magnetic_field': 1.0  # T
        }

        self.material_parameters = self._initialize_material_parameters()

    def _initialize_material_parameters(self) -> Dict:
        """Initialize material-specific parameters"""
        return {
            'Si': {
                'lattice_constant': 5.431,  # Å
                'spin_orbit_coupling': 44e-6,  # eV
                'g_factor': 2.0,
                'hyperfine_coupling': 117e6,  # Hz
                'debye_temperature': 645  # K
            },
            'Ge': {
                'lattice_constant': 5.658,
                'spin_orbit_coupling': 296e-6,
                'g_factor': 1.6,
                'hyperfine_coupling': 7.4e6,
                'debye_temperature': 374
            },
            'GaAs': {
                'lattice_constant': 5.653,
                'spin_orbit_coupling': 341e-6,
                'g_factor': -0.44,
                'hyperfine_coupling': 85e6,
                'debye_temperature': 360
            },
            'C': {
                'lattice_constant': 3.567,
                'spin_orbit_coupling': 6e-6,
                'g_factor': 2.003,
                'hyperfine_coupling': 2.7e6,
                'debye_temperature': 2230
            }
        }

    def generate_composition(self) -> Tuple[str, str, Dict]:
        """Generate quantum material composition with defects"""
        host_type = np.random.choice(list(self.host_materials.keys()))
        host = np.random.choice(self.host_materials[host_type])

        defect_type = np.random.choice(list(self.defect_types.keys()))
        defect = np.random.choice(self.defect_types[defect_type])

        distribution = {
            'mean_depth': np.random.uniform(5, 100),  # nm
            'spread': np.random.uniform(1, 10),  # nm
            'lateral_size': np.random.uniform(10, 1000),  # nm
            'concentration': 10 ** np.random.uniform(14, 18),  # cm^-3
            'strain': np.random.uniform(0, 0.02)  # fractional
        }

        return host, defect, distribution

    def calculate_coherence_properties(
            self,
            host: str,
            defect: str,
            temperature: float,
            magnetic_field: float,
            distribution: Dict
    ) -> Dict[str, float]:
        """Calculate coherence-related properties"""
        params = self.material_parameters.get(host, self.material_parameters['Si'])

        # Phonon-induced decoherence
        gamma_phonon = self._calculate_phonon_decoherence(
            temperature, params['debye_temperature'])

        # Magnetic noise decoherence
        gamma_magnetic = self._calculate_magnetic_decoherence(
            magnetic_field, params['g_factor'])

        # Strain-induced decoherence
        gamma_strain = self._calculate_strain_decoherence(
            distribution['strain'], params['spin_orbit_coupling'])

        # Total decoherence rate
        gamma_total = gamma_phonon + gamma_magnetic + gamma_strain

        # Calculate coherence times
        T2_intrinsic = 1 / gamma_total
        T1 = T2_intrinsic * np.random.uniform(1.5, 2.5)
        T2_star = T2_intrinsic * np.random.uniform(0.1, 0.5)

        return {
            'T1': T1,
            'T2': T2_intrinsic,
            'T2_star': T2_star,
            'gamma_phonon': gamma_phonon,
            'gamma_magnetic': gamma_magnetic,
            'gamma_strain': gamma_strain
        }

    def _calculate_phonon_decoherence(
            self,
            temperature: float,
            debye_temp: float
    ) -> float:
        """Calculate phonon-induced decoherence rate"""
        # Direct process
        gamma_direct = temperature * (temperature / debye_temp) ** 5

        # Raman process
        gamma_raman = temperature ** 7 * (temperature / debye_temp) ** 2

        # Orbach process
        delta_E = 10  # meV
        gamma_orbach = np.exp(-delta_E / (self.quantum_constants['k_boltzmann'] * temperature))

        return 1e3 * (gamma_direct + gamma_raman + gamma_orbach)

    def _calculate_magnetic_decoherence(
            self,
            magnetic_field: float,
            g_factor: float
    ) -> float:
        """Calculate magnetic field induced decoherence"""
        # Zeeman splitting
        E_zeeman = g_factor * const.physical_constants['Bohr magneton'][0] * magnetic_field

        # Fluctuation strength
        delta_B = 1e-9 * magnetic_field  # T

        gamma = (g_factor * const.physical_constants['Bohr magneton'][0] * delta_B /
                 self.quantum_constants['hbar'])

        return gamma

    def _calculate_strain_decoherence(
            self,
            strain: float,
            spin_orbit: float
    ) -> float:
        """Calculate strain-induced decoherence"""
        return strain * spin_orbit / self.quantum_constants['hbar']

    def calculate_qubit_properties(
            self,
            host: str,
            defect: str,
            coherence: Dict[str, float],
            temperature: float
    ) -> Dict[str, float]:
        """Calculate qubit-specific properties"""
        params = self.material_parameters.get(host, self.material_parameters['Si'])

        # Transition frequency with temperature dependence
        f01_base = np.random.uniform(*self.property_ranges['transition_frequency'])
        f01 = f01_base * (1 - 1e-4 * (temperature / params['debye_temperature']) ** 4)

        # Anharmonicity
        alpha = np.random.uniform(*self.property_ranges['anharmonicity'])
        alpha *= (1 - temperature / (20 * self.quantum_constants['base_temperature']))

        # Coupling strength
        g_base = np.random.uniform(*self.property_ranges['coupling_strength'])
        g = g_base * np.sqrt(1 - temperature / (10 * self.quantum_constants['base_temperature']))

        # Quality factors
        Q_internal = coherence['T1'] * f01 * 2 * np.pi
        Q_coupling = g / (2 * np.pi * f01)
        Q_total = 1 / (1 / Q_internal + 1 / Q_coupling)

        # Frequency stability
        df_thermal = f01 * 1e-6 * temperature  # Hz/K
        df_charge = g * 1e-3  # Hz/e

        return {
            'frequency_01': f01,
            'anharmonicity': alpha,
            'coupling_strength': g,
            'Q_internal': Q_internal,
            'Q_coupling': Q_coupling,
            'Q_total': Q_total,
            'df_thermal': df_thermal,
            'df_charge': df_charge
        }

    def calculate_gate_properties(
            self,
            coherence: Dict[str, float],
            qubit: Dict[str, float],
            temperature: float
    ) -> Dict[str, float]:
        """Calculate gate operation properties"""
        # Single-qubit gate time
        t_gate_1q = 1 / (2 * qubit['coupling_strength'])

        # Two-qubit gate time with thermal effects
        t_gate_2q = t_gate_1q * np.random.uniform(5, 10)
        t_gate_2q *= (1 + temperature / (5 * self.quantum_constants['base_temperature']))

        # Calculate fidelities with thermal and charge noise effects
        F_1q_base = np.exp(-t_gate_1q / coherence['T2'])
        F_2q_base = np.exp(-t_gate_2q / coherence['T2'])

        # Temperature-dependent error rates
        thermal_error = temperature / (10 * self.quantum_constants['base_temperature'])
        charge_error = 1e-4 * np.random.uniform(0.5, 2.0)

        F_1q = F_1q_base * (1 - thermal_error - charge_error)
        F_2q = F_2q_base * (1 - 2 * thermal_error - 2 * charge_error)

        # Readout properties with realistic imperfections
        t_readout = np.random.uniform(100e-9, 500e-9)  # s
        F_readout_base = np.random.uniform(*self.property_ranges['readout_fidelity'])
        F_readout = F_readout_base * (1 - thermal_error)

        return {
            't_gate_1q': t_gate_1q,
            't_gate_2q': t_gate_2q,
            'F_1q': F_1q,
            'F_2q': F_2q,
            't_readout': t_readout,
            'F_readout': F_readout,
            'thermal_error': thermal_error,
            'charge_error': charge_error
        }

    def calculate_noise_properties(
            self,
            temperature: float,
            coherence: Dict[str, float],
            qubit: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate noise and error properties"""
        # Thermal noise
        kT = self.quantum_constants['k_boltzmann'] * temperature
        hf = self.quantum_constants['h_planck'] * qubit['frequency_01']
        thermal_population = 1 / (np.exp(hf / kT) - 1)

        # Charge noise with temperature dependence
        charge_noise_base = 1e-6  # e/√Hz
        charge_noise = charge_noise_base * np.exp(-temperature / 2)

        # Flux noise with temperature and frequency dependence
        flux_noise_base = 1e-6  # Φ₀/√Hz
        flux_noise = flux_noise_base * (1 + np.log(qubit['frequency_01'] / 1e9))

        # Critical current noise
        ic_noise_base = 1e-6
        ic_noise = ic_noise_base * np.sqrt(1 + (temperature / 0.1) ** 2)

        # Photon shot noise
        shot_noise = np.sqrt(thermal_population)

        # Total noise including correlations
        total_noise = np.sqrt(
            charge_noise ** 2 + flux_noise ** 2 + ic_noise ** 2 +
            2 * charge_noise * flux_noise * 0.3  # Correlation term
        )

        return {
            'thermal_population': thermal_population,
            'charge_noise': charge_noise,
            'flux_noise': flux_noise,
            'ic_noise': ic_noise,
            'shot_noise': shot_noise,
            'total_noise': total_noise
        }

    def calculate_scalability_metrics(
            self,
            host: str,
            defect: str,
            distribution: Dict[str, float],
            coherence: Dict[str, float],
            gate: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate metrics related to scalability"""
        # Basic fabrication yield with material complexity
        params = self.material_parameters.get(host, self.material_parameters['Si'])
        complexity_factor = len(defect.split('_')) * 0.9
        strain_factor = np.exp(-distribution['strain'] * 10)

        basic_yield = np.random.uniform(0.6, 0.95)
        fabrication_yield = (basic_yield * complexity_factor *
                             strain_factor * np.exp(-distribution['mean_depth'] / 1000))

        # Qubit density with spacing constraints
        min_spacing = 4 * distribution['lateral_size']  # nm
        density = 1e14 / (min_spacing ** 2)  # qubits/cm²

        # Connectivity analysis
        max_connections = int(4 * np.pi * distribution['lateral_size'])
        coupling_strength = np.exp(-min_spacing / 500)  # Normalized
        achieved_connections = int(max_connections * coupling_strength)

        # Crosstalk calculations
        direct_crosstalk = 0.01 * np.exp(-min_spacing / 100)
        indirect_crosstalk = 0.005 * coupling_strength ** 2
        total_crosstalk = direct_crosstalk + indirect_crosstalk

        # Integration compatibility
        process_compatibility = {
            'Si': 0.95,
            'Ge': 0.85,
            'GaAs': 0.75,
            'C': 0.90
        }.get(host, 0.7)

        thermal_budget = {
            'Si': 1200,  # °C
            'Ge': 900,
            'GaAs': 800,
            'C': 1500
        }.get(host, 1000)

        integration_score = (process_compatibility *
                             fabrication_yield *
                             (1 - total_crosstalk) *
                             (thermal_budget / 1500))

        return {
            'fabrication_yield': fabrication_yield,
            'qubit_density': density,
            'connectivity': achieved_connections,
            'direct_crosstalk': direct_crosstalk,
            'indirect_crosstalk': indirect_crosstalk,
            'total_crosstalk': total_crosstalk,
            'process_compatibility': process_compatibility,
            'thermal_budget': thermal_budget,
            'integration_score': integration_score
        }
    def generate_dataset(self, n_samples: int = 1000000) -> pd.DataFrame:
        """Generate comprehensive quantum materials dataset"""
        data = []

        for _ in tqdm(range(n_samples)):
            # Generate base material properties
            host, defect, distribution = self.generate_composition()
            temperature = np.random.uniform(0.010, 0.100)  # K
            magnetic_field = np.random.uniform(0, 2.0)  # T

            # Calculate all properties
            coherence = self.calculate_coherence_properties(
                host, defect, temperature, magnetic_field, distribution)

            qubit = self.calculate_qubit_properties(
                host, defect, coherence, temperature)

            gate = self.calculate_gate_properties(
                coherence, qubit, temperature)

            noise = self.calculate_noise_properties(
                temperature, coherence, qubit)

            scalability = self.calculate_scalability_metrics(
                host, defect, distribution, coherence, gate)

            # Environmental sensitivities
            sensitivities = self.calculate_environmental_sensitivities(
                host, defect, temperature, magnetic_field)

            # Combined properties
            sample = {
                'host_material': host,
                'defect_type': defect,
                'temperature': temperature,
                'magnetic_field': magnetic_field,
                'mean_depth': distribution['mean_depth'],
                'lateral_size': distribution['lateral_size'],
                'strain': distribution['strain'],
                **coherence,
                **qubit,
                **gate,
                **noise,
                **scalability,
                **sensitivities
            }

            data.append(sample)

        return pd.DataFrame(data)

    def calculate_environmental_sensitivities(
            self,
            host: str,
            defect: str,
            temperature: float,
            magnetic_field: float
    ) -> Dict[str, float]:
        """Calculate sensitivities to environmental parameters"""
        params = self.material_parameters.get(host, self.material_parameters['Si'])

        # Temperature sensitivity
        dT = 0.001  # K
        temp_sensitivity = abs(
            self._calculate_coherence_change(
                host, defect, temperature + dT, magnetic_field) -
            self._calculate_coherence_change(
                host, defect, temperature - dT, magnetic_field)
        ) / (2 * dT)

        # Magnetic field sensitivity
        dB = 0.001  # T
        field_sensitivity = abs(
            self._calculate_coherence_change(
                host, defect, temperature, magnetic_field + dB) -
            self._calculate_coherence_change(
                host, defect, temperature, magnetic_field - dB)
        ) / (2 * dB)

        # Electric field sensitivity
        electric_sensitivity = params['g_factor'] * 1e-3  # Hz/(V/m)

        # Strain sensitivity
        strain_sensitivity = params['spin_orbit_coupling'] * 1e3  # Hz/strain

        return {
            'temperature_sensitivity': temp_sensitivity,
            'magnetic_sensitivity': field_sensitivity,
            'electric_sensitivity': electric_sensitivity,
            'strain_sensitivity': strain_sensitivity
        }

    def _calculate_coherence_change(
            self,
            host: str,
            defect: str,
            temperature: float,
            magnetic_field: float
    ) -> float:
        """Helper function for sensitivity calculations"""
        distribution = {
            'mean_depth': 50,
            'spread': 5,
            'lateral_size': 100,
            'strain': 0.01
        }
        coherence = self.calculate_coherence_properties(
            host, defect, temperature, magnetic_field, distribution)
        return coherence['T2']

    def generate_time_evolution(
            self,
            sample: pd.Series,
            duration: float = 1e-3,
            n_points: int = 1000
    ) -> Dict[str, np.ndarray]:
        """Generate time evolution data for a material sample"""
        time = np.linspace(0, duration, n_points)

        # Coherence decay with realistic noise
        noise_amplitude = 0.05
        noise = noise_amplitude * np.random.randn(n_points)

        T1_decay = np.exp(-time / sample['T1']) + noise
        T2_decay = np.exp(-time / sample['T2']) + noise

        # Rabi oscillations with detuning and decay
        rabi_freq = sample['coupling_strength']
        detuning = np.random.normal(0, rabi_freq * 0.01)
        rabi = (np.cos(2 * np.pi * (rabi_freq + detuning) * time) *
                np.exp(-time / sample['T2']) + noise)

        # Ramsey fringes with frequency drift
        base_detuning = 1e6  # Hz
        drift_rate = 1e3  # Hz/s
        effective_detuning = base_detuning + drift_rate * time
        ramsey = (np.cos(2 * np.pi * effective_detuning * time) *
                  np.exp(-time / sample['T2_star']) + noise)

        # Echo sequence
        echo_decay = np.exp(-time / (2 * sample['T2'])) + noise

        return {
            'time': time,
            'T1_decay': T1_decay,
            'T2_decay': T2_decay,
            'rabi': rabi,
            'ramsey': ramsey,
            'echo': echo_decay
        }

# Fix the indentation - move this outside the class definition
if __name__ == "__main__":
    # Initialize generator
    generator = QuantumMaterialGenerator()

    # Generate large dataset for training
    print("Generating main dataset...")
    df = generator.generate_dataset(n_samples=1000000)

    # Save to CSV
    df.to_csv("quantum_materials.csv", index=False)

    # Generate time evolution data for selected samples
    print("Generating time evolution data...")
    samples = df.sample(n=100)
    evolution_data = []

    for _, sample in samples.iterrows():
        evolution = generator.generate_time_evolution(sample)
        evolution_data.append(evolution)

    # Save time evolution data
    np.save("quantum_evolution_data.npy", evolution_data)

    print("Dataset generation complete.")
    print(f"Total samples: {len(df)}")
    print(f"Time evolution samples: {len(evolution_data)}")

    # Basic statistics
    print("\nDataset Statistics:")
    print(f"Average T2 coherence time: {df['T2'].mean():.2e} s")
    print(f"Average gate fidelity: {df['F_1q'].mean():.4f}")
    print(f"Average qubit density: {df['qubit_density'].mean():.2e} qubits/cm²")