# integrated_main.py
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pyquil import Program
from circuits import create_4gt13_92, create_4mod5_v1_22, create_decod24_v2_43, create_ising_model_10, create_ising_model_13, create_ising_model_7, create_mod5mils_65
from sabre_tools.sabre import SABRE
from sabre_tools.circuit_preprocess import get_initial_mapping, get_distance_matrix, preprocess_input_circuit

class QuantumBenchmark:
    """Integrated benchmarking system for SABRE heuristics"""
    
    def __init__(self, warmup_runs=3, measured_runs=10):
        self.warmup_runs = warmup_runs
        self.measured_runs = measured_runs
        self.results = {}
        
        self.benchmarks = {
            '4mod5-v1_22': create_4mod5_v1_22,
            'mod5mils_65': create_mod5mils_65,
            'decod24-v2_43': create_decod24_v2_43,
            '4gt13_92': create_4gt13_92,
            'ising_7': create_ising_model_7,
            'ising_10': create_ising_model_10,
            'ising_13': create_ising_model_13,
        }
    
    def run_benchmark(self, circuit_func):
        """Execute benchmark with timing measurements"""
        heuristics = ['original', 'critical', 'entanglement', 'lookahead']
        
        # Get circuit and its specific coupling graph
        circuit, coupling_graph = circuit_func()
        circuit_name = circuit_func.__name__.replace('create_', '')
        
        # Visualize coupling graph
        self._visualize_coupling_graph(coupling_graph, circuit_name)
        
        self.results[circuit_name] = {}
        
        for heuristic in heuristics:
            print(f"\nBenchmarking {heuristic} on {circuit_name}...")
            timings = []
            gate_metrics = []
            
            # Warmup runs
            for _ in range(self.warmup_runs):
                self._execute_single_run(circuit.copy(), coupling_graph, heuristic)
            
            # Timed runs
            for _ in range(self.measured_runs):
                start_time = time.perf_counter()
                metrics = self._execute_single_run(circuit.copy(), coupling_graph, heuristic)
                timings.append(time.perf_counter() - start_time)
                gate_metrics.append(metrics)
            
            # Store results
            self.results[circuit_name][heuristic] = {
                'time_stats': self._compute_stats(timings),
                'gate_stats': self._compute_gate_stats(gate_metrics)
            }
        
        return self.results[circuit_name]
    
    def _visualize_coupling_graph(self, graph, title):
        """Visualize the coupling graph for a circuit"""
        plt.figure(figsize=(6, 4))
        nx.draw(graph, with_labels=True, node_color='lightblue', 
                node_size=800, font_weight='bold')
        plt.title(f"Coupling Graph for {title}")
        plt.tight_layout()
        plt.show()
    
    def _execute_single_run(self, circuit, coupling_graph, heuristic):
        """Single execution of benchmark"""
        initial_mapping = get_initial_mapping(circuit, coupling_graph)
        distance_matrix = get_distance_matrix(coupling_graph)
        
        sabre_proc = SABRE(distance_matrix, coupling_graph, heuristic=heuristic)
        temp_mapping = initial_mapping.copy()
        temp_circuit = circuit.copy()
        
        # Forward-backward-forward passes
        for _ in range(3):
            front_layer, dag = preprocess_input_circuit(temp_circuit)
            final_program, final_mapping = sabre_proc.execute_sabre_algorithm(
                front_layer_gates=front_layer,
                qubit_mapping=temp_mapping,
                circuit_dag=dag
            )
            temp_circuit = Program(list(reversed(temp_circuit.instructions)))
            temp_mapping = final_mapping.copy()
        
        # Collect metrics
        forbidden = sabre_proc.rewiring_correctness(final_program, final_mapping)
        return {
            'original_gates': sabre_proc.cnot_count(circuit),
            'final_gates': sabre_proc.cnot_count(final_program),
            'success': not bool(forbidden),
            'swap_count': (sabre_proc.cnot_count(final_program) - sabre_proc.cnot_count(circuit)) // 3
        }
    
    def _compute_stats(self, timings):
        return {
            'mean': np.mean(timings),
            'std': np.std(timings),
            'min': min(timings),
            'max': max(timings)
        }
    
    def _compute_gate_stats(self, metrics):
        return {
            'original': metrics[0]['original_gates'],
            'final_mean': np.mean([m['final_gates'] for m in metrics]),
            'swap_mean': np.mean([m['swap_count'] for m in metrics]),
            'success_rate': np.mean([m['success'] for m in metrics])
        }
    
    def run_all_benchmarks(self):
        """Execute all benchmarks with all heuristics"""
        for circuit_func in self.benchmarks.values():
            self.run_benchmark(circuit_func)
    
    def generate_report(self, save_path=None):
        """Generate comprehensive performance report with improved visualizations"""
        # Prepare data
        circuit_names = list(self.results.keys())
        heuristics = list(next(iter(self.results.values())).keys())
        
        # Set up color palette
        colors = plt.cm.tab10.colors  # Using a built-in colormap for distinct colors
        bar_width = 0.8 / len(circuit_names)  # Dynamic width based on number of circuits
        
        # Timing comparison plot - Grouped Bar Chart
        plt.figure(figsize=(14, 7))
        for i, circuit_name in enumerate(circuit_names):
            data = self.results[circuit_name]
            times = [d['time_stats']['mean'] for d in data.values()]
            errors = [d['time_stats']['std'] for d in data.values()]
            
            x = np.arange(len(heuristics))
            plt.bar(x + i*bar_width, times, width=bar_width, 
                    yerr=errors, color=colors[i], alpha=0.8, 
                    label=circuit_name, capsize=5)
        
        plt.xlabel('Heuristic')
        plt.ylabel('Time (s)')
        plt.title('Execution Time Comparison by Circuit and Heuristic')
        plt.xticks(x + bar_width*(len(circuit_names)-1)/2, heuristics)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_timing.png", bbox_inches='tight', dpi=300)
        plt.show()
        
        # Gate overhead plot - Grouped Bar Chart
        plt.figure(figsize=(14, 7))
        for i, circuit_name in enumerate(circuit_names):
            data = self.results[circuit_name]
            overhead = [d['gate_stats']['final_mean'] - d['gate_stats']['original'] for d in data.values()]
            
            x = np.arange(len(heuristics))
            plt.bar(x + i*bar_width, overhead, width=bar_width, 
                    color=colors[i], alpha=0.8, label=circuit_name)
        
        plt.xlabel('Heuristic')
        plt.ylabel('Additional Gates')
        plt.title('Circuit Overhead Comparison by Circuit and Heuristic')
        plt.xticks(x + bar_width*(len(circuit_names)-1)/2, heuristics)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_overhead.png", bbox_inches='tight', dpi=300)
        plt.show()
        
        # Success rate plot - Grouped Bar Chart
        plt.figure(figsize=(14, 7))
        for i, circuit_name in enumerate(circuit_names):
            data = self.results[circuit_name]
            success = [d['gate_stats']['success_rate']*100 for d in data.values()]
            
            x = np.arange(len(heuristics))
            plt.bar(x + i*bar_width, success, width=bar_width, 
                    color=colors[i], alpha=0.8, label=circuit_name)
        
        plt.xlabel('Heuristic')
        plt.ylabel('Success Rate (%)')
        plt.title('Success Rate Comparison by Circuit and Heuristic')
        plt.xticks(x + bar_width*(len(circuit_names)-1)/2, heuristics)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_success.png", bbox_inches='tight', dpi=300)
        plt.show()
        
        # Print numerical results in a more readable format
        print("\n=== Detailed Benchmark Results ===")
        for circuit, data in self.results.items():
            print(f"\nCircuit: {circuit}")
            print("-" * (len(circuit) + 10))
            print(f"{'Heuristic':<15} {'Time (s)':<12} {'Gates':<15} {'SWAPs':<10} {'Success':<10}")
            print("-" * 60)
            for heuristic, metrics in data.items():
                print(f"{heuristic:<15} {metrics['time_stats']['mean']:.4f} ± {metrics['time_stats']['std']:.4f}  "
                    f"{metrics['gate_stats']['original']}→{metrics['gate_stats']['final_mean']:.1f}  "
                    f"{metrics['gate_stats']['swap_mean']:<10.1f} "
                    f"{metrics['gate_stats']['success_rate']:.1%}")

if __name__ == "__main__":
    # Initialize benchmark system
    benchmark = QuantumBenchmark(warmup_runs=3, measured_runs=10)
    
    # Run all benchmarks
    print("Starting benchmark execution...")

    benchmark.run_all_benchmarks()
    
    # Generate comprehensive report
    benchmark.generate_report("sabre_benchmark_results")
    
    print("\nBenchmarking completed!")