import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pyquil import Program
from circuits import create_4gt13_92, create_4mod5_v1_22, create_decod24_v2_43, create_ising_model_10, create_ising_model_13, create_ising_model_16, create_mod5mils_65, create_qft_10
from sabre_tools.heuristics import critical_path_heuristic, decay_weighted_distance_heuristic, decay_weighted_distance_heuristic_upgraded, entanglement_aware_heuristic, hybrid_lexi_heuristic, lookahead_window_heuristic
from sabre_tools.sabre import SABRE
from sabre_tools.circuit_preprocess import get_distance_matrix, preprocess_input_circuit
from sabre_tools.state_analysis import build_interaction_graph, compute_initial_layout

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
            'ising_10': create_ising_model_10,
            'ising_13': create_ising_model_13,
            'ising_16': create_ising_model_16,
            'qft_10': create_qft_10,
        }

        self.heuristics = {
            'original': decay_weighted_distance_heuristic,
            'decay_weighted': decay_weighted_distance_heuristic_upgraded,
            'critical': critical_path_heuristic,
            'entanglement': entanglement_aware_heuristic,
            'lookahead': lookahead_window_heuristic,
            'hybrid_lexi': hybrid_lexi_heuristic,
        }
    
    def run_benchmark(self, circuit_func):
        """Execute benchmark with timing measurements"""
        # heuristics = ['original', 'hybrid_lexi', 'decay_weighted', 'critical', 'entanglement', 'lookahead']
        heuristics = ['original', 'hybrid_lexi'] # Some heuristics may take too long to run
        
        circuit, coupling_graph = circuit_func()
        circuit_name = circuit_func.__name__.replace('create_', '')
        
        self._visualize_coupling_graph(coupling_graph, circuit_name)
        
        self.results[circuit_name] = {}
        
        for heuristic in heuristics:
            print(f"\nBenchmarking {heuristic} on {circuit_name}...")
            timings = []
            gate_metrics = []
            
            for _ in range(self.warmup_runs):
                self._execute_single_run(circuit.copy(), coupling_graph, self.heuristics[heuristic])
            
            for _ in range(self.measured_runs):
                start_time = time.perf_counter()
                metrics = self._execute_single_run(circuit.copy(), coupling_graph, self.heuristics[heuristic])
                timings.append(time.perf_counter() - start_time)
                gate_metrics.append(metrics)
            
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

    def _cancel_adjacent_swaps(self, prog: Program) -> Program:
        new_instr = []
        i = 0
        while i < len(prog.instructions):
            inst = prog.instructions[i]
            if inst.name == 'SWAP' and i+1 < len(prog.instructions):
                nxt = prog.instructions[i+1]
                if nxt.name == 'SWAP' and set(inst.qubits) == set(nxt.qubits):
                    i += 2          # skip both – they cancel
                    continue
            new_instr.append(inst)
            i += 1
        return Program(new_instr)
    
    def _execute_single_run(self, circuit, coupling_graph, heuristic):
        """Single execution of benchmark"""
        inter_graph = build_interaction_graph(circuit)
        initial_mapping = compute_initial_layout(inter_graph, coupling_graph)
        distance_matrix = get_distance_matrix(coupling_graph)
        device = nx.Graph(coupling_graph.edges())
        
        sabre_proc = SABRE(distance_matrix, coupling_graph, heuristic=heuristic)
        temp_mapping = initial_mapping.copy()
        temp_circuit = circuit.copy()
        
        for _ in range(2):
            front_layer, dag = preprocess_input_circuit(temp_circuit)
            final_program, final_mapping = sabre_proc.execute_sabre_algorithm(
                front_layer_gates=front_layer,
                qubit_mapping=temp_mapping,
                circuit_dag=dag
            )
            temp_circuit = Program(list(reversed(final_program.instructions)))
            temp_mapping = final_mapping.copy()
        
        final_program = self._cancel_adjacent_swaps(final_program)
        
        success = True
        for instr in final_program:
            if len(instr.qubits) == 2:
                q1, q2 = instr.qubits
                if not device.has_edge(q1.index, q2.index):
                    success = False
                    break
        return {
            'original_gates': sabre_proc.cnot_count(circuit),
            'final_gates': sabre_proc.cnot_count(final_program),
            'success': success,
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
        circuit_names = list(self.results.keys())
        heuristics = list(next(iter(self.results.values())).keys())
        
        colors = plt.cm.tab10.colors 
        bar_width = 0.8 / len(circuit_names) 
        
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

        records = []
        for circuit_name, heur_data in self.results.items():
            for heuristic_name, metrics in heur_data.items():
                records.append({
                    "Circuit":    circuit_name,
                    "Heuristic":  heuristic_name,
                    "Time (s)":   metrics["time_stats"]["mean"],
                })

        df = (
            pd.DataFrame(records)
            .pivot(index="Circuit", columns="Heuristic", values="Time (s)")
            .sort_index()
            .sort_index(axis=1)
        )

        print(df.to_string(float_format="%.4f"))

        fig, ax = plt.subplots(figsize=(1.8 + 1.2*len(df.columns),
                                        1.0 + 0.4*len(df)))
        ax.axis("off")

        tbl = ax.table(
            cellText=df.round(4).values,
            rowLabels=df.index,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.3)

        plt.title("Mean compile time (s)", fontweight="bold", pad=12)
        fig.tight_layout()

        plt.show()

        if save_path:
            png_file = f"{save_path}_table_time.png"
            fig.savefig(png_file, dpi=300, bbox_inches="tight")

                

if __name__ == "__main__":
    benchmark = QuantumBenchmark(warmup_runs=3, measured_runs=10)

    print("Starting benchmark execution...")
    
    benchmark.run_all_benchmarks()
    
    benchmark.generate_report("sabre_benchmark_results")
    
    print("\nBenchmarking completed!")