#!/usr/bin/env python3
# Phase 1.3: Performance Analysis and Visualization
# This script generates the roofline model and performance comparison plots
# The roofline model helps us understand if kernels are memory-bound or compute-bound

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import re

def parse_gpu_specs_file(filename):
    """Parse GPU specs from the generated file
    Reads peak TFLOPS and bandwidth from gpu_specs.txt for roofline calculations"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Extract peak TFLOPS
        fp32_match = re.search(r'Estimated FP32 Peak TFLOPS:\s+([\d.]+)', content)
        fp16_match = re.search(r'TensorCore FP16 Peak TFLOPS:\s+([\d.]+)', content)
        bw_match = re.search(r'Peak Memory Bandwidth:\s+([\d.]+)', content)
        
        if fp32_match and bw_match:
            result = {
                'peak_gflops_fp32': float(fp32_match.group(1)) * 1000,  # Convert TFLOPS to GFLOPS
                'peak_gflops_fp16': float(fp16_match.group(1)) * 1000 if fp16_match else 0,
                'peak_bandwidth_gb_s': float(bw_match.group(1))
            }
            print(f"Parsed GPU specs from file:")
            print(f"  FP32 Peak: {result['peak_gflops_fp32']:.1f} GFLOPS")
            print(f"  FP16 Peak: {result['peak_gflops_fp16']:.1f} GFLOPS")
            print(f"  Bandwidth: {result['peak_bandwidth_gb_s']:.1f} GB/s")
            return result
    except Exception as e:
        print(f"Warning: Could not parse GPU specs file: {e}")
    
    return None

def load_benchmark_data(filename):
    """Load benchmark results from CSV"""
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)

def calculate_arithmetic_intensity(M, N, K, bytes_per_element):
    """
    Calculate arithmetic intensity for GEMM
    AI = FLOPS / Bytes
    
    Arithmetic Intensity tells us if a workload is memory-bound or compute-bound:
    - High AI (> ridge point): compute-bound (limited by FLOPS)
    - Low AI (< ridge point): memory-bound (limited by bandwidth)
    
    For GEMM:
    - FLOPS = 2*M*N*K (each element of C requires K multiply-adds = 2 ops)
    - Bytes = (M*K + K*N + M*N) * bytes_per_element (read A, read B, write C)
    """
    flops = 2.0 * M * N * K
    bytes_transferred = (M * K + K * N + M * N) * bytes_per_element
    return flops / bytes_transferred

def plot_roofline(df, gpu_specs, output_file):
    """Create roofline plot with custom visualization
    
    The roofline model shows:
    - X-axis: Arithmetic Intensity (FLOPS/Byte)
    - Y-axis: Performance (GFLOPS)
    - Roofline curve: Theoretical peak performance at each AI
      - Left side (low AI): Memory-bound region (slope = bandwidth)
      - Right side (high AI): Compute-bound region (flat = peak FLOPS)
    - Points: Actual kernel performance
    
    Kernels below the roofline have optimization opportunities!
    """
    # Custom figure with different style
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    
    # Calculate arithmetic intensity for each benchmark
    # FP16 uses 2 bytes per element, FP32 uses 4 bytes
    df['bytes_per_element'] = df['DType'].apply(lambda x: 2 if x == 'FP16' else 4)
    df['AI'] = df.apply(lambda row: calculate_arithmetic_intensity(
        row['M'], row['N'], row['K'], row['bytes_per_element']), axis=1)
    
    # Roofline parameters from GPU specs
    peak_bandwidth = gpu_specs['peak_bandwidth_gb_s']
    peak_gflops_fp32 = gpu_specs['peak_gflops_fp32']
    peak_gflops_fp16 = gpu_specs['peak_gflops_fp16']
    
    # Create roofline curves
    # Generate range of AI values for plotting the roofline
    ai_min = max(0.1, df['AI'].min() * 0.5)
    ai_max = min(1000, df['AI'].max() * 2)
    ai_range = np.logspace(np.log10(ai_min), np.log10(ai_max), 1000)
    
    # FP32 roofline: min of memory-bound and compute-bound limits
    # Memory-bound: Performance = AI * Bandwidth (slope)
    # Compute-bound: Performance = Peak FLOPS (flat line)
    memory_bound_fp32 = ai_range * peak_bandwidth
    compute_bound_fp32 = np.full_like(ai_range, peak_gflops_fp32)
    roofline_fp32 = np.minimum(memory_bound_fp32, compute_bound_fp32)
    
    # Plot rooflines with custom styling
    ax.loglog(ai_range, roofline_fp32, color='#2c3e50', linewidth=3, 
              label='FP32 Theoretical Peak', zorder=1, linestyle='-')
    
    if peak_gflops_fp16 > 0:
        memory_bound_fp16 = ai_range * peak_bandwidth
        compute_bound_fp16 = np.full_like(ai_range, peak_gflops_fp16)
        roofline_fp16 = np.minimum(memory_bound_fp16, compute_bound_fp16)
        ax.loglog(ai_range, roofline_fp16, color='#8e44ad', linewidth=3, 
                 label='FP16 TensorCore Theoretical Peak', zorder=1, linestyle='--')
    
    # Plot benchmark results with custom colors and styling
    kernels = df['Kernel'].unique()
    # Custom color scheme - different from typical examples
    colors = {
        'Lab1_Tiled': '#c0392b',  # Dark red
        'cuBLAS_SGEMM': '#16a085',  # Teal green
        'cuBLAS_HGEMM_TensorCore': '#2980b9'  # Blue
    }
    markers = {
        'Lab1_Tiled': 'D',  # Diamond
        'cuBLAS_SGEMM': 'P',  # Plus (filled)
        'cuBLAS_HGEMM_TensorCore': 'X'  # X marker
    }
    
    for kernel in kernels:
        kernel_df = df[df['Kernel'] == kernel]
        color = colors.get(kernel, 'gray')
        marker = markers.get(kernel, 'o')
        ax.loglog(kernel_df['AI'], kernel_df['GFLOPS'],
                 marker=marker, color=color, markersize=11,
                 linestyle='', label=kernel.replace('_', ' '), 
                 alpha=0.75, markeredgewidth=2,
                 markeredgecolor='white', zorder=3, linewidth=2)
    
    # Calculate and annotate ridge points
    ridge_point_fp32 = peak_gflops_fp32 / peak_bandwidth
    ax.axvline(x=ridge_point_fp32, color='#34495e', linestyle=':', 
               alpha=0.5, linewidth=2, label=f'FP32 Ridge Point ({ridge_point_fp32:.1f} FLOPS/Byte)')
    
    if peak_gflops_fp16 > 0:
        ridge_point_fp16 = peak_gflops_fp16 / peak_bandwidth
        ax.axvline(x=ridge_point_fp16, color='#7d3c98', linestyle=':', 
                  alpha=0.5, linewidth=2, label=f'FP16 Ridge Point ({ridge_point_fp16:.1f} FLOPS/Byte)')
    
    # Add shaded regions to show memory-bound vs compute-bound
    # Memory-bound region (left of ridge point)
    ax.axvspan(ai_min, ridge_point_fp32, alpha=0.1, color='orange', 
              label='Memory-Bound Region')
    
    # Labels and formatting with custom style
    ax.set_xlabel('Arithmetic Intensity (FLOPS per Byte)', 
                 fontsize=13, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Performance (GFLOPS)', 
                 fontsize=13, fontweight='bold', color='#2c3e50')
    ax.set_title('GEMM Performance Roofline Analysis\nPhase 1 Baseline Results', 
                fontsize=15, fontweight='bold', pad=15, color='#2c3e50')
    
    # Custom grid
    ax.grid(True, which='major', linestyle='-', alpha=0.3, linewidth=0.8, color='gray')
    ax.grid(True, which='minor', linestyle=':', alpha=0.2, linewidth=0.5, color='lightgray')
    
    # Custom legend
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95, 
             edgecolor='black', fancybox=True, shadow=True)
    
    # Set axis limits
    y_min = max(1, df['GFLOPS'].min() * 0.5)
    y_max = peak_gflops_fp32 * 2 if peak_gflops_fp16 == 0 else peak_gflops_fp16 * 1.5
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(ai_min, ai_max)
    
    # Custom tick styling
    ax.tick_params(axis='both', which='major', labelsize=10, 
                  colors='#2c3e50', width=1.5)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
    # Add text annotation for key insights
    ax.text(0.02, 0.98, 'Higher is Better\nPoints below roofline indicate\noptimization opportunities', 
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Roofline plot saved to {output_file}")
    plt.close()

def plot_performance_comparison(df, output_file):
    """Plot performance comparison across different matrix sizes with custom styling"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('white')
    
    # Filter square matrices
    square_df = df[(df['M'] == df['N']) & (df['N'] == df['K'])].copy()
    
    if len(square_df) > 0:
        kernels = square_df['Kernel'].unique()
        # Custom color scheme - different from standard
        colors = {
            'Lab1_Tiled': '#8b0000',  # Dark red
            'cuBLAS_SGEMM': '#006400',  # Dark green
            'cuBLAS_HGEMM_TensorCore': '#00008b'  # Dark blue
        }
        markers = {
            'Lab1_Tiled': 'v',  # Down triangle
            'cuBLAS_SGEMM': 's',  # Square
            'cuBLAS_HGEMM_TensorCore': '*'  # Star
        }
        linestyles = {
            'Lab1_Tiled': '-',
            'cuBLAS_SGEMM': '--',
            'cuBLAS_HGEMM_TensorCore': '-.'
        }
        
        # Plot 1: GFLOPS vs Matrix Size
        for kernel in kernels:
            kernel_df = square_df[square_df['Kernel'] == kernel].sort_values('M')
            ax1.plot(kernel_df['M'], kernel_df['GFLOPS'],
                    marker=markers.get(kernel, 'o'), 
                    label=kernel.replace('_', ' '), 
                    color=colors.get(kernel, 'gray'),
                    linewidth=3, markersize=9, markeredgewidth=2, 
                    markeredgecolor='white', linestyle=linestyles.get(kernel, '-'),
                    alpha=0.85)
        
        ax1.set_xlabel('Matrix Dimension (M = N = K)', fontsize=12, fontweight='bold', color='#2c3e50')
        ax1.set_ylabel('Performance (GFLOPS)', fontsize=12, fontweight='bold', color='#2c3e50')
        ax1.set_title('GEMM Throughput Scaling Analysis', fontsize=13, fontweight='bold', pad=10, color='#2c3e50')
        ax1.legend(fontsize=10, framealpha=0.9, loc='upper left')
        ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        ax1.tick_params(axis='both', which='major', labelsize=10, colors='#2c3e50')
        
        # Plot 2: Efficiency vs Matrix Size
        cublas_fp32 = square_df[square_df['Kernel'] == 'cuBLAS_SGEMM']
        if len(cublas_fp32) > 0:
            for kernel in kernels:
                if kernel == 'cuBLAS_SGEMM':
                    continue
                kernel_df = square_df[square_df['Kernel'] == kernel].sort_values('M')
                
                efficiency = []
                sizes = []
                for _, row in kernel_df.iterrows():
                    ref = cublas_fp32[cublas_fp32['M'] == row['M']]
                    if len(ref) > 0:
                        eff = (row['GFLOPS'] / ref['GFLOPS'].values[0]) * 100
                        efficiency.append(eff)
                        sizes.append(row['M'])
                
                ax2.plot(sizes, efficiency, 
                        marker=markers.get(kernel, 'o'), 
                        label=f"{kernel.replace('_', ' ')} relative to cuBLAS FP32",
                        color=colors.get(kernel, 'gray'), 
                        linewidth=3, markersize=9,
                        markeredgewidth=2, markeredgecolor='white',
                        linestyle=linestyles.get(kernel, '-'), alpha=0.85)
        
        ax2.set_xlabel('Matrix Dimension (M = N = K)', fontsize=12, fontweight='bold', color='#2c3e50')
        ax2.set_ylabel('Relative Performance (% of cuBLAS FP32)', fontsize=12, fontweight='bold', color='#2c3e50')
        ax2.set_title('Performance Efficiency Comparison', fontsize=13, fontweight='bold', pad=10, color='#2c3e50')
        ax2.legend(fontsize=10, framealpha=0.9, loc='best')
        ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)
        ax2.set_xscale('log', base=2)
        ax2.axhline(y=100, color='#006400', linestyle='--', alpha=0.6, linewidth=2.5, 
                   label='100% Baseline (cuBLAS FP32)')
        ax2.tick_params(axis='both', which='major', labelsize=10, colors='#2c3e50')
        
        # Add annotation for key insight
        ax2.text(0.02, 0.98, 'Values > 100% indicate\nbetter than cuBLAS FP32', 
                transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Performance comparison saved to {output_file}")
    plt.close()

def generate_analysis_report(df, gpu_specs, output_file):
    """Generate detailed analysis report
    Phase 1.3 Deliverable: Performance comparison report
    Includes efficiency calculations and gap analysis"""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GEMM PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("GPU SPECIFICATIONS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Peak FP32 Performance: {gpu_specs['peak_gflops_fp32']:.2f} GFLOPS\n")
        f.write(f"Peak FP16 TensorCore Performance: {gpu_specs['peak_gflops_fp16']:.2f} GFLOPS\n")
        f.write(f"Peak Memory Bandwidth: {gpu_specs['peak_bandwidth_gb_s']:.2f} GB/s\n")
        if gpu_specs['peak_gflops_fp16'] > 0:
            f.write(f"TensorCore Speedup (theoretical): {gpu_specs['peak_gflops_fp16']/gpu_specs['peak_gflops_fp32']:.1f}x\n\n")
        
        # Calculate arithmetic intensity
        df['bytes_per_element'] = df['DType'].apply(lambda x: 2 if x == 'FP16' else 4)
        df['AI'] = df.apply(lambda row: calculate_arithmetic_intensity(
            row['M'], row['N'], row['K'], row['bytes_per_element']), axis=1)
        
        # Performance summary by kernel
        f.write("PERFORMANCE SUMMARY BY KERNEL:\n")
        f.write("-" * 40 + "\n")
        for kernel in df['Kernel'].unique():
            kernel_df = df[df['Kernel'] == kernel]
            f.write(f"\n{kernel}:\n")
            f.write(f"  Average GFLOPS: {kernel_df['GFLOPS'].mean():.2f}\n")
            f.write(f"  Max GFLOPS: {kernel_df['GFLOPS'].max():.2f}\n")
            f.write(f"  Min GFLOPS: {kernel_df['GFLOPS'].min():.2f}\n")
            f.write(f"  Average Bandwidth: {kernel_df['Bandwidth_GB_s'].mean():.2f} GB/s\n")
            f.write(f"  Average AI: {kernel_df['AI'].mean():.2f} FLOPS/Byte\n")
            
            # Calculate efficiency
            if 'FP16' in kernel_df['DType'].values[0]:
                peak = gpu_specs['peak_gflops_fp16']
            else:
                peak = gpu_specs['peak_gflops_fp32']
            
            if peak > 0:
                efficiency = (kernel_df['GFLOPS'].mean() / peak) * 100
                f.write(f"  Average Efficiency: {efficiency:.2f}% of peak\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Analysis report saved to {output_file}")

def main():
    # Try to parse GPU specs from file
    gpu_specs = parse_gpu_specs_file('results/gpu_specs.txt')
    
    # Fallback to V100 specs if parsing fails
    if gpu_specs is None:
        print("Using fallback V100 GPU specs")
        gpu_specs = {
            'peak_gflops_fp32': 15670.0,   # V100: 15.67 TFLOPS
            'peak_gflops_fp16': 112000.0,  # V100: 112.0 TFLOPS
            'peak_bandwidth_gb_s': 898.0   # V100: 898.0 GB/s
        }
    
    # Load benchmark data
    df = load_benchmark_data('results/benchmark_results.csv')
    
    print(f"\nLoaded {len(df)} benchmark results")
    print(f"Kernels tested: {', '.join(df['Kernel'].unique())}")
    print(f"Matrix sizes: {len(df[df['Kernel'] == df['Kernel'].iloc[0]])}")
    
    # Generate visualizations and reports
    print("\nGenerating analysis...")
    plot_roofline(df, gpu_specs, 'results/roofline_plot.png')
    plot_performance_comparison(df, 'results/performance_comparison.png')
    generate_analysis_report(df, gpu_specs, 'results/analysis_report.txt')
    
    print("\nAnalysis complete!")
    print("Generated files:")
    print("  - results/roofline_plot.png")
    print("  - results/performance_comparison.png")
    print("  - results/analysis_report.txt")

if __name__ == '__main__':
    main()