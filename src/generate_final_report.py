#!/usr/bin/env python3
"""
Phase 4: Final Comprehensive Analysis and Report Generation
Generates final performance summary, comparisons, and project report
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_benchmark_results(csv_path):
    """Load benchmark results from CSV"""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return None
    
    df = pd.read_csv(csv_path)
    return df

def generate_performance_summary(df, output_file):
    """Generate comprehensive performance summary"""
    
    # Filter for key square matrix sizes
    key_sizes = [1024, 2048, 4096, 8192]
    df_square = df[(df['M'] == df['N']) & (df['N'] == df['K']) & (df['M'].isin(key_sizes))]
    
    # Get cuBLAS TensorCore as baseline
    cublas_tc = df_square[df_square['kernel_name'] == 'cuBLAS_HGEMM_TensorCore']
    
    if len(cublas_tc) == 0:
        print("Warning: cuBLAS TensorCore results not found")
        return
    
    # Calculate efficiency for each kernel
    results = []
    for kernel in df_square['kernel_name'].unique():
        kernel_data = df_square[df_square['kernel_name'] == kernel]
        
        for size in key_sizes:
            size_data = kernel_data[kernel_data['M'] == size]
            if len(size_data) > 0:
                cublas_baseline = cublas_tc[cublas_tc['M'] == size]
                if len(cublas_baseline) > 0:
                    efficiency = (size_data['gflops'].values[0] / cublas_baseline['gflops'].values[0] * 100)
                    results.append({
                        'kernel': kernel,
                        'size': size,
                        'gflops': size_data['gflops'].values[0],
                        'time_ms': size_data['time_ms'].values[0],
                        'efficiency_%': efficiency
                    })
    
    results_df = pd.DataFrame(results)
    
    # Generate summary by kernel
    summary = results_df.groupby('kernel').agg({
        'gflops': ['mean', 'max', 'min'],
        'efficiency_%': ['mean', 'max', 'min']
    }).round(2)
    
    # Write summary
    with open(output_file, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('FINAL PERFORMANCE SUMMARY\n')
        f.write('=' * 80 + '\n\n')
        f.write('Key Matrix Sizes: 1024, 2048, 4096, 8192 (square matrices)\n\n')
        
        f.write('Performance by Kernel (GFLOPS):\n')
        f.write('-' * 80 + '\n')
        for kernel in results_df['kernel'].unique():
            kernel_data = results_df[results_df['kernel'] == kernel]
            f.write(f"\n{kernel}:\n")
            for _, row in kernel_data.iterrows():
                f.write(f"  {row['size']}×{row['size']}×{row['size']}: "
                       f"{row['gflops']:.2f} GFLOPS "
                       f"({row['efficiency_%']:.2f}% of cuBLAS TensorCore)\n")
        
        f.write('\n' + '=' * 80 + '\n')
        f.write('Summary Statistics:\n')
        f.write('-' * 80 + '\n')
        f.write(str(summary) + '\n\n')
        
        # Calculate overall efficiency
        our_kernels = ['Lab2_TensorCore', 'Lab2_TensorCore_Optimized', 'Lab2_TensorCore_LargeTile']
        our_results = results_df[results_df['kernel'].isin(our_kernels)]
        if len(our_results) > 0:
            avg_efficiency = our_results['efficiency_%'].mean()
            f.write(f"Average Efficiency of Our TensorCore Kernels: {avg_efficiency:.2f}%\n")
            f.write(f"Target: 40-60% of cuBLAS TensorCore\n")
            f.write(f"Status: {'✓ ACHIEVED' if avg_efficiency >= 40 else '✗ NOT ACHIEVED'}\n")
        
        f.write('\n' + '=' * 80 + '\n')
    
    print(f"✓ Performance summary written to {output_file}")

def generate_comparison_table(df, output_file):
    """Generate comparison table for report"""
    
    # Filter for 4096×4096×4096 (representative large size)
    df_4096 = df[(df['M'] == 4096) & (df['N'] == 4096) & (df['K'] == 4096)]
    
    if len(df_4096) == 0:
        print("Warning: 4096×4096×4096 results not found")
        return
    
    # Get cuBLAS TensorCore baseline
    cublas_tc = df_4096[df_4096['kernel_name'] == 'cuBLAS_HGEMM_TensorCore']
    if len(cublas_tc) == 0:
        print("Warning: cuBLAS TensorCore results not found")
        return
    
    baseline_gflops = cublas_tc['gflops'].values[0]
    
    with open(output_file, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('PERFORMANCE COMPARISON (4096×4096×4096)\n')
        f.write('=' * 80 + '\n\n')
        f.write(f"Baseline (cuBLAS TensorCore): {baseline_gflops:.2f} GFLOPS\n\n")
        f.write(f"{'Kernel':<35} {'GFLOPS':>12} {'Efficiency':>12} {'Time (ms)':>12}\n")
        f.write('-' * 80 + '\n')
        
        for _, row in df_4096.iterrows():
            efficiency = (row['gflops'] / baseline_gflops * 100)
            f.write(f"{row['kernel_name']:<35} "
                   f"{row['gflops']:>12.2f} "
                   f"{efficiency:>11.2f}% "
                   f"{row['time_ms']:>12.4f}\n")
        
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
    
    csv_path = results_dir / "benchmark_results.csv"
    
    # Load data
    print("Loading benchmark results...")
    df = load_benchmark_results(csv_path)
    if df is None:
        return
    
    print(f"Loaded {len(df)} benchmark results")
    print()
    
    # Generate reports
    print("Generating performance summary...")
    generate_performance_summary(df, final_dir / "performance_summary.txt")
    print()
    
    print("Generating comparison table...")
    generate_comparison_table(df, final_dir / "comparison_table.txt")
    print()
    
    print("=" * 80)
    print("Final report generation complete!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  • {final_dir / 'performance_summary.txt'}")
    print(f"  • {final_dir / 'comparison_table.txt'}")

if __name__ == "__main__":
    main()

