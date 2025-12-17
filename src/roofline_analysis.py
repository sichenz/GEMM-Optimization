#!/usr/bin/env python3
# Generate roofline plots and performance analysis
# Roofline model shows if kernels are memory-bound or compute-bound

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import re

def parse_gpu_specs_file(filename):
    """Parse GPU specs from file"""
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
        
        # Filter to only essential kernels (clean up redundant ones)
        essential_kernels = [
            'Lab1_Tiled',
            'Lab2_TensorCore',  # Baseline
            'Lab2_TensorCore_Optimized',  # Optimization attempt
            'Lab2_TensorCore_Balanced',  # Best performer
            'cuBLAS_SGEMM',
            'cuBLAS_HGEMM_TensorCore'
        ]
        
        if 'Kernel' in df.columns:
            df = df[df['Kernel'].isin(essential_kernels)]
            print(f"Filtered to essential kernels: {len(df)} results")
        
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)

def calculate_arithmetic_intensity(M, N, K, bytes_per_element):
    """Calculate arithmetic intensity: FLOPS / Bytes
    High AI = compute-bound, Low AI = memory-bound
    For GEMM: FLOPS = 2*M*N*K, Bytes = (M*K + K*N + M*N) * bytes_per_element
    """
    flops = 2.0 * M * N * K
    bytes_transferred = (M * K + K * N + M * N) * bytes_per_element
    return flops / bytes_transferred

def plot_roofline(df, gpu_specs, output_file):
    """
    Option C (NVIDIA-style): separate plots by precision.
    - FP32 plot: only FP32 results + FP32 peak horizontal line
    - FP16 plot: only FP16 results + FP16 TensorCore peak horizontal line

    X-axis: matrix size (square matrices only, M=N=K)
    Y-axis: achieved GFLOPS
    """

    import os

    # --- helper to save with suffix ---
    def with_suffix(path, suffix):
        root, ext = os.path.splitext(path)
        return f"{root}_{suffix}{ext}"

    # Filter to square matrices only
    square_df = df[(df['M'] == df['N']) & (df['N'] == df['K'])].copy()
    if len(square_df) == 0:
        print("Warning: No square matrices found (M == N == K). Skipping plots.")
        return

    peak_fp32 = float(gpu_specs.get('peak_gflops_fp32', 0.0))
    peak_fp16 = float(gpu_specs.get('peak_gflops_fp16', 0.0))

    # Common styling maps (keep your existing look)
    colors = {
        'Lab1_Tiled': '#c0392b',
        'cuBLAS_SGEMM': '#16a085',
        'cuBLAS_HGEMM_TensorCore': '#2980b9',
        'Lab2_TensorCore': 'gray',
        'Lab2_TensorCore_Optimized': 'gray',
        'Lab2_TensorCore_Balanced': 'gray',
    }
    markers = {
        'Lab1_Tiled': 'D',
        'cuBLAS_SGEMM': 'P',
        'cuBLAS_HGEMM_TensorCore': 'X',
        'Lab2_TensorCore': 'o',
        'Lab2_TensorCore_Optimized': 'o',
        'Lab2_TensorCore_Balanced': 'o',
    }
    linestyles = {
        'Lab2_TensorCore': '-',
        'Lab2_TensorCore_Optimized': '--',
        'Lab2_TensorCore_Balanced': '-.',
    }

    def make_plot(plot_df, peak_line, title, out_path, line_label):
        if len(plot_df) == 0:
            print(f"Warning: No data for {title}. Skipping {out_path}.")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')

        kernels = plot_df['Kernel'].unique()
        for kernel in kernels:
            kdf = plot_df[plot_df['Kernel'] == kernel].sort_values('M')
            ax.plot(
                kdf['M'], kdf['GFLOPS'],
                marker=markers.get(kernel, 'o'),
                color=colors.get(kernel, 'gray'),
                linestyle=linestyles.get(kernel, '-'),
                linewidth=2, markersize=10,
                markeredgewidth=2, markeredgecolor='white',
                alpha=0.85,
                label=kernel.replace('_', ' ')
            )

        # Hardware peak line for this precision
        if peak_line > 0:
            ax.axhline(
                y=peak_line,
                color='#2c3e50',
                linestyle='--',
                linewidth=3,
                alpha=0.85,
                label=line_label
            )

        ax.set_xlabel('Matrix Size (M = N = K)', fontsize=13, fontweight='bold', color='#2c3e50')
        ax.set_ylabel('Achieved Performance (GFLOPS)', fontsize=13, fontweight='bold', color='#2c3e50')
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15, color='#2c3e50')

        ax.grid(True, which='major', linestyle='-', alpha=0.3, linewidth=0.8, color='gray')
        ax.grid(True, which='minor', linestyle=':', alpha=0.2, linewidth=0.5, color='lightgray')

        # Log2 x-axis is typical for GEMM sizes (powers of 2)
        ax.set_xscale('log', base=2)

        # Nice limits
        y_min = max(1, plot_df['GFLOPS'].min() * 0.6)
        y_max = max(plot_df['GFLOPS'].max() * 1.15, (peak_line * 1.15 if peak_line > 0 else 0))
        ax.set_ylim(y_min, y_max)

        x_min = max(1, plot_df['M'].min())
        x_max = plot_df['M'].max()
        ax.set_xlim(x_min, x_max)

        ax.legend(loc='best', fontsize=10, framealpha=0.95,
                  edgecolor='black', fancybox=True, shadow=True)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {out_path}")

    # --- FP32 plot ---
    fp32_df = square_df[square_df['DType'] == 'FP32'].copy()
    make_plot(
        plot_df=fp32_df,
        peak_line=peak_fp32,
        title='GEMM FP32 Performance vs Matrix Size (Square Matrices)',
        out_path=with_suffix(output_file, 'fp32'),
        line_label=f"FP32 Peak ({peak_fp32:.0f} GFLOPS)" if peak_fp32 > 0 else "FP32 Peak"
    )

    # --- FP16 / TensorCore plot ---
    fp16_df = square_df[square_df['DType'] == 'FP16'].copy()
    make_plot(
        plot_df=fp16_df,
        peak_line=peak_fp16,
        title='GEMM FP16 TensorCore Performance vs Matrix Size (Square Matrices)',
        out_path=with_suffix(output_file, 'fp16'),
        line_label=f"FP16 TensorCore Peak ({peak_fp16:.0f} GFLOPS)" if peak_fp16 > 0 else "FP16 TensorCore Peak"
    )

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
    """Generate analysis report with efficiency calculations"""
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
    
    # Load benchmark data (try final results first, then fallback)
    import os
    csv_path = 'results/final/benchmark_results.csv'
    if not os.path.exists(csv_path):
        csv_path = 'results/benchmark_results.csv'
    df = load_benchmark_data(csv_path)
    
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

        