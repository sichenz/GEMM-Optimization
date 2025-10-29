"""
Phase 1 Analysis and Visualization
Generates performance reports and roofline plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class PerformanceAnalyzer:
    def __init__(self, results_file='benchmark_results.csv', 
                 roofline_file='roofline_data.csv'):
        self.results = pd.read_csv(results_file)
        self.roofline = pd.read_csv(roofline_file)
        
    def plot_performance_comparison(self):
        """Compare Lab-1 vs cuBLAS performance"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Group by matrix size
        lab1_data = self.results[self.results['kernel'] == 'Lab-1 Tiled']
        cublas_data = self.results[self.results['kernel'] == 'cuBLAS']
        
        # Create size labels
        lab1_data['size'] = lab1_data['m'].astype(str) + '×' + \
                            lab1_data['n'].astype(str) + '×' + \
                            lab1_data['k'].astype(str)
        cublas_data['size'] = cublas_data['m'].astype(str) + '×' + \
                              cublas_data['n'].astype(str) + '×' + \
                              cublas_data['k'].astype(str)
        
        # Plot 1: GFLOPS comparison
        ax1 = axes[0, 0]
        x = np.arange(len(lab1_data))
        width = 0.35
        ax1.bar(x - width/2, lab1_data['gflops'], width, label='Lab-1', alpha=0.8)
        ax1.bar(x + width/2, cublas_data['gflops'], width, label='cuBLAS', alpha=0.8)
        ax1.set_xlabel('Matrix Size')
        ax1.set_ylabel('GFLOPS')
        ax1.set_title('Performance Comparison: GFLOPS')
        ax1.set_xticks(x)
        ax1.set_xticklabels(lab1_data['size'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Compute efficiency
        ax2 = axes[0, 1]
        ax2.plot(lab1_data['size'], lab1_data['compute_eff'], 
                marker='o', label='Lab-1', linewidth=2)
        ax2.plot(cublas_data['size'], cublas_data['compute_eff'], 
                marker='s', label='cuBLAS', linewidth=2)
        ax2.set_xlabel('Matrix Size')
        ax2.set_ylabel('Compute Efficiency (%)')
        ax2.set_title('Compute Efficiency vs Matrix Size')
        ax2.set_xticklabels(lab1_data['size'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Speedup
        ax3 = axes[1, 0]
        speedup = cublas_data['gflops'].values / lab1_data['gflops'].values
        ax3.bar(lab1_data['size'], speedup, alpha=0.8, color='coral')
        ax3.axhline(y=1.0, color='r', linestyle='--', label='Parity')
        ax3.set_xlabel('Matrix Size')
        ax3.set_ylabel('Speedup (cuBLAS / Lab-1)')
        ax3.set_title('cuBLAS Speedup over Lab-1')
        ax3.set_xticklabels(lab1_data['size'], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Time comparison
        ax4 = axes[1, 1]
        ax4.semilogy(lab1_data['size'], lab1_data['time_ms'], 
                    marker='o', label='Lab-1', linewidth=2)
        ax4.semilogy(cublas_data['size'], cublas_data['time_ms'], 
                    marker='s', label='cuBLAS', linewidth=2)
        ax4.set_xlabel('Matrix Size')
        ax4.set_ylabel('Time (ms, log scale)')
        ax4.set_title('Execution Time Comparison')
        ax4.set_xticklabels(lab1_data['size'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: performance_comparison.png")
        plt.close()
        
    def plot_roofline(self):
        """Generate roofline plot with actual performance points"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot roofline
        ai = self.roofline['arithmetic_intensity']
        peak = self.roofline['peak_gflops']
        memory_bound = self.roofline['memory_bound_gflops']
        
        ax.loglog(ai, peak, 'k--', linewidth=2, label='Compute Bound')
        ax.loglog(ai, memory_bound, 'b-', linewidth=2, label='Memory Bound')
        
        # Plot actual performance points
        lab1_data = self.results[self.results['kernel'] == 'Lab-1 Tiled']
        cublas_data = self.results[self.results['kernel'] == 'cuBLAS']
        
        ax.scatter(lab1_data['arithmetic_intensity'], 
                  lab1_data['gflops'],
                  s=100, c='red', marker='o', alpha=0.7, 
                  label='Lab-1 Tiled', edgecolors='black', linewidth=1.5)
        
        ax.scatter(cublas_data['arithmetic_intensity'], 
                  cublas_data['gflops'],
                  s=100, c='green', marker='s', alpha=0.7,
                  label='cuBLAS', edgecolors='black', linewidth=1.5)
        
        # Annotate some key points
        for idx in range(0, len(lab1_data), max(1, len(lab1_data)//5)):
            row = lab1_data.iloc[idx]
            size_str = f"{row['m']}×{row['n']}"
            ax.annotate(size_str, 
                       xy=(row['arithmetic_intensity'], row['gflops']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Arithmetic Intensity (FLOPS/Byte)', fontsize=12)
        ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
        ax.set_title('Roofline Model: Lab-1 vs cuBLAS', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig('roofline_plot.png', dpi=300, bbox_inches='tight')
        print("Saved: roofline_plot.png")
        plt.close()
        
    def generate_report(self):
        """Generate detailed performance report"""
        lab1_data = self.results[self.results['kernel'] == 'Lab-1 Tiled']
        cublas_data = self.results[self.results['kernel'] == 'cuBLAS']
        
        # Merge for comparison
        merged = pd.merge(
            lab1_data, cublas_data,
            on=['m', 'n', 'k'],
            suffixes=('_lab1', '_cublas')
        )
        
        merged['speedup'] = merged['gflops_cublas'] / merged['gflops_lab1']
        merged['gap_gflops'] = merged['gflops_cublas'] - merged['gflops_lab1']
        
        report = []
        report.append("=" * 80)
        report.append("PHASE 1: PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("### Summary Statistics ###")
        report.append(f"Lab-1 Tiled GEMM:")
        report.append(f"  Average GFLOPS: {lab1_data['gflops'].mean():.2f}")
        report.append(f"  Peak GFLOPS: {lab1_data['gflops'].max():.2f}")
        report.append(f"  Average Efficiency: {lab1_data['compute_eff'].mean():.2f}%")
        report.append(f"  Peak Efficiency: {lab1_data['compute_eff'].max():.2f}%")
        report.append("")
        
        report.append(f"cuBLAS:")
        report.append(f"  Average GFLOPS: {cublas_data['gflops'].mean():.2f}")
        report.append(f"  Peak GFLOPS: {cublas_data['gflops'].max():.2f}")
        report.append(f"  Average Efficiency: {cublas_data['compute_eff'].mean():.2f}%")
        report.append(f"  Peak Efficiency: {cublas_data['compute_eff'].max():.2f}%")
        report.append("")
        
        report.append("### Performance Gap Analysis ###")
        report.append(f"Average Speedup (cuBLAS/Lab-1): {merged['speedup'].mean():.2f}x")
        report.append(f"Maximum Speedup: {merged['speedup'].max():.2f}x")
        report.append(f"Minimum Speedup: {merged['speedup'].min():.2f}x")
        report.append(f"Average Gap: {merged['gap_gflops'].mean():.2f} GFLOPS")
        report.append("")
        
        # Detailed per-size analysis
        report.append("### Detailed Performance by Matrix Size ###")
        report.append(f"{'Size':<20} {'Lab-1':>10} {'cuBLAS':>10} {'Speedup':>10} {'Gap':>10}")
        report.append("-" * 70)
        
        for _, row in merged.iterrows():
            size = f"{row['m']}×{row['n']}×{row['k']}"
            report.append(
                f"{size:<20} "
                f"{row['gflops_lab1']:>10.2f} "
                f"{row['gflops_cublas']:>10.2f} "
                f"{row['speedup']:>10.2f}x "
                f"{row['gap_gflops']:>10.2f}"
            )
        
        report.append("")
        report.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        with open('performance_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print("\nReport saved to: performance_report.txt")
        
    def analyze_bottlenecks(self):
        """Analyze performance bottlenecks"""
        lab1_data = self.results[self.results['kernel'] == 'Lab-1 Tiled']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Arithmetic intensity vs efficiency
        ax1 = axes[0, 0]
        scatter = ax1.scatter(lab1_data['arithmetic_intensity'], 
                            lab1_data['compute_eff'],
                            c=lab1_data['m'] * lab1_data['n'] * lab1_data['k'],
                            cmap='viridis', s=100, alpha=0.7)
        ax1.set_xlabel('Arithmetic Intensity (FLOPS/Byte)')
        ax1.set_ylabel('Compute Efficiency (%)')
        ax1.set_title('Efficiency vs Arithmetic Intensity')
        plt.colorbar(scatter, ax=ax1, label='Total FLOPs')
        ax1.grid(True, alpha=0.3)
        
        # Memory bandwidth utilization
        ax2 = axes[0, 1]
        ax2.scatter(lab1_data['m'] * lab1_data['n'] * lab1_data['k'],
                   lab1_data['memory_eff'],
                   s=100, alpha=0.7, color='coral')
        ax2.set_xlabel('Problem Size (M×N×K)')
        ax2.set_ylabel('Memory Efficiency (%)')
        ax2.set_title('Memory Efficiency vs Problem Size')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Performance scaling
        ax3 = axes[1, 0]
        problem_sizes = lab1_data['m'] * lab1_data['n'] * lab1_data['k']
        ax3.loglog(problem_sizes, lab1_data['gflops'], 
                  marker='o', linestyle='-', linewidth=2)
        ax3.set_xlabel('Problem Size (M×N×K)')
        ax3.set_ylabel('GFLOPS')
        ax3.set_title('Performance Scaling')
        ax3.grid(True, alpha=0.3)
        
        # Time breakdown
        ax4 = axes[1, 1]
        sizes = lab1_data['m'].astype(str) + '×' + \
                lab1_data['n'].astype(str) + '×' + \
                lab1_data['k'].astype(str)
        ax4.bar(range(len(lab1_data)), lab1_data['time_ms'], alpha=0.7)
        ax4.set_xlabel('Matrix Configuration')
        ax4.set_ylabel('Time (ms)')
        ax4.set_title('Execution Time by Configuration')
        ax4.set_xticks(range(len(lab1_data)))
        ax4.set_xticklabels(sizes, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bottleneck_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved: bottleneck_analysis.png")
        plt.close()

def main():
    print("=== Phase 1: Performance Analysis ===\n")
    
    # Check if results file exists
    if not Path('benchmark_results.csv').exists():
        print("Error: benchmark_results.csv not found!")
        print("Please run benchmark_gemm first.")
        return
    
    analyzer = PerformanceAnalyzer()
    
    print("Generating performance comparison plots...")
    analyzer.plot_performance_comparison()
    
    print("Generating roofline plot...")
    analyzer.plot_roofline()
    
    print("Analyzing bottlenecks...")
    analyzer.analyze_bottlenecks()
    
    print("Generating detailed report...")
    analyzer.generate_report()
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("  - performance_comparison.png")
    print("  - roofline_plot.png")
    print("  - bottleneck_analysis.png")
    print("  - performance_report.txt")

if __name__ == '__main__':
    main()