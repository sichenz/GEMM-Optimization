#!/usr/bin/env python3
"""
Phase 4: Final Comprehensive Analysis and Report Generation
Generates final performance summary, comparisons, and project report
Uses only standard library (no pandas required)
"""

import os
import csv
from pathlib import Path
from collections import defaultdict

def load_benchmark_results(csv_path):
    """Load benchmark results from CSV"""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return None
    
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'kernel_name': row['kernel_name'],
                'dtype': row['dtype'],
                'M': int(row['M']),
                'N': int(row['N']),
                'K': int(row['K']),
                'time_ms': float(row['time_ms']),
                'gflops': float(row['gflops']),
                'bandwidth_gb_s': float(row['bandwidth_gb_s'])
            })
    return results

def generate_performance_summary(results, output_file):
    """Generate comprehensive performance summary"""
    
    # Filter for key square matrix sizes
    key_sizes = [1024, 2048, 4096, 8192]
    square_results = [r for r in results if r['M'] == r['N'] == r['K'] and r['M'] in key_sizes]
    
    # Get cuBLAS TensorCore baseline for each size
    cublas_baselines = {}
    for size in key_sizes:
        for r in square_results:
            if r['kernel_name'] == 'cuBLAS_HGEMM_TensorCore' and r['M'] == size:
                cublas_baselines[size] = r['gflops']
                break
    
    # Calculate efficiency for each kernel
    kernel_data = defaultdict(list)
    for r in square_results:
        size = r['M']
        if size in cublas_baselines:
            efficiency = (r['gflops'] / cublas_baselines[size] * 100)
            kernel_data[r['kernel_name']].append({
                'size': size,
                'gflops': r['gflops'],
                'time_ms': r['time_ms'],
                'efficiency': efficiency
            })
    
    # Write summary
    with open(output_file, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('FINAL PERFORMANCE SUMMARY\n')
        f.write('=' * 80 + '\n\n')
        f.write('Key Matrix Sizes: 1024, 2048, 4096, 8192 (square matrices)\n\n')
        
        f.write('Performance by Kernel (GFLOPS):\n')
        f.write('-' * 80 + '\n')
        for kernel in sorted(kernel_data.keys()):
            data = kernel_data[kernel]
            f.write(f"\n{kernel}:\n")
            for entry in sorted(data, key=lambda x: x['size']):
                f.write(f"  {entry['size']}×{entry['size']}×{entry['size']}: "
                       f"{entry['gflops']:.2f} GFLOPS "
                       f"({entry['efficiency']:.2f}% of cuBLAS TensorCore)\n")
        
        f.write('\n' + '=' * 80 + '\n')
        f.write('Summary Statistics:\n')
        f.write('-' * 80 + '\n')
        
        # Calculate statistics
        for kernel in sorted(kernel_data.keys()):
            data = kernel_data[kernel]
            gflops_values = [e['gflops'] for e in data]
            efficiency_values = [e['efficiency'] for e in data]
            if gflops_values:
                f.write(f"\n{kernel}:\n")
                f.write(f"  GFLOPS: mean={sum(gflops_values)/len(gflops_values):.2f}, "
                       f"max={max(gflops_values):.2f}, min={min(gflops_values):.2f}\n")
                f.write(f"  Efficiency: mean={sum(efficiency_values)/len(efficiency_values):.2f}%, "
                       f"max={max(efficiency_values):.2f}%, min={min(efficiency_values):.2f}%\n")
        
        # Calculate overall efficiency for our kernels
        our_kernels = ['Lab2_TensorCore', 'Lab2_TensorCore_Optimized', 'Lab2_TensorCore_LargeTile']
        our_efficiencies = []
        for kernel in our_kernels:
            if kernel in kernel_data:
                our_efficiencies.extend([e['efficiency'] for e in kernel_data[kernel]])
        
        if our_efficiencies:
            avg_efficiency = sum(our_efficiencies) / len(our_efficiencies)
            f.write('\n' + '=' * 80 + '\n')
            f.write(f"Average Efficiency of Our TensorCore Kernels: {avg_efficiency:.2f}%\n")
            f.write(f"Target: 40-60% of cuBLAS TensorCore\n")
            f.write(f"Status: {'✓ ACHIEVED' if avg_efficiency >= 40 else '✗ NOT ACHIEVED'}\n")
        
        f.write('\n' + '=' * 80 + '\n')
    
    print(f"✓ Performance summary written to {output_file}")

def generate_comparison_table(results, output_file):
    """Generate comparison table for report"""
    
    # Filter for 4096×4096×4096
    results_4096 = [r for r in results if r['M'] == 4096 and r['N'] == 4096 and r['K'] == 4096]
    
    if not results_4096:
        print("Warning: 4096×4096×4096 results not found")
        return
    
    # Get cuBLAS TensorCore baseline
    baseline_gflops = None
    for r in results_4096:
        if r['kernel_name'] == 'cuBLAS_HGEMM_TensorCore':
            baseline_gflops = r['gflops']
            break
    
    if baseline_gflops is None:
        print("Warning: cuBLAS TensorCore results not found")
        return
    
    with open(output_file, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('PERFORMANCE COMPARISON (4096×4096×4096)\n')
        f.write('=' * 80 + '\n\n')
        f.write(f"Baseline (cuBLAS TensorCore): {baseline_gflops:.2f} GFLOPS\n\n")
        f.write(f"{'Kernel':<35} {'GFLOPS':>12} {'Efficiency':>12} {'Time (ms)':>12}\n")
        f.write('-' * 80 + '\n')
        
        for r in sorted(results_4096, key=lambda x: x['kernel_name']):
            efficiency = (r['gflops'] / baseline_gflops * 100)
            f.write(f"{r['kernel_name']:<35} "
                   f"{r['gflops']:>12.2f} "
                   f"{efficiency:>11.2f}% "
                   f"{r['time_ms']:>12.4f}\n")
        
        f.write('\n' + '=' * 80 + '\n')
    
    print(f"✓ Comparison table written to {output_file}")

def main():
    """Main function"""
    print("=" * 80)
    print("Phase 4: Final Report Generation")
    print("=" * 80)
    print()
    
    # Paths
    results_dir = Path("results")
    final_dir = results_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = final_dir / "benchmark_results.csv"
    
    # Load data
    print("Loading benchmark results...")
    results = load_benchmark_results(csv_path)
    if results is None:
        return
    
    print(f"Loaded {len(results)} benchmark results")
    print()
    
    # Generate reports
    print("Generating performance summary...")
    generate_performance_summary(results, final_dir / "performance_summary.txt")
    print()
    
    print("Generating comparison table...")
    generate_comparison_table(results, final_dir / "comparison_table.txt")
    print()
    
    print("=" * 80)
    print("Final report generation complete!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  • {final_dir / 'performance_summary.txt'}")
    print(f"  • {final_dir / 'comparison_table.txt'}")

if __name__ == "__main__":
    main()
