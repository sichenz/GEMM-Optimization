#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import re

def parse_gpu_specs_file(filename):
    """Parse GPU specs from the generated file"""
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
    """
    flops = 2.0 * M * N * K
    bytes_transferred = (M * K + K * N + M * N) * bytes_per_element
    return flops / bytes_transferred

def plot_roofline(df, gpu_specs, output_file):
    """
    Plot GEMM performance vs matrix size for square matrices, with a single
    hardware peak GFLOPS line.
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    # Use only square matrices: M = N = K
    square_df = df[(df['M'] == df['N']) & (df['N'] == df['K'])].copy()
    if len(square_df) == 0:
        print("Warning: No square matrices found in the benchmark data. Skipping roofline plot.")
        plt.close()
        return

    # Get unique kernels and define colors/markers
    kernels = square_df['Kernel'].unique()
    colors = {
        'Lab1_Tiled': '#e74c3c',
        'cuBLAS_SGEMM': '#27ae60',
        'cuBLAS_HGEMM_TensorCore': '#3498db'
    }
    markers = {
        'Lab1_Tiled': 'o',
        'cuBLAS_SGEMM': 's',
        'cuBLAS_HGEMM_TensorCore': '^'
    }

    # Plot GFLOPS vs matrix size for each kernel
    for kernel in kernels:
        kernel_df = square_df[square_df['Kernel'] == kernel].sort_values('M')
        ax.plot(
            kernel_df['M'],
            kernel_df['GFLOPS'],
            marker=markers.get(kernel, 'o'),
            linestyle='-',
            linewidth=2.5,
            markersize=8,
            markeredgewidth=1.5,
            markeredgecolor='white',
            color=colors.get(kernel, 'gray'),
            label=kernel,
            alpha=0.9,
        )

    # Determine hardware peak GFLOPS (single limit line)
    peak_fp32 = gpu_specs.get('peak_gflops_fp32', 0.0)
    peak_fp16 = gpu_specs.get('peak_gflops_fp16', 0.0)
    hardware_peak = max(peak_fp32, peak_fp16)

    if hardware_peak > 0:
        ax.axhline(
            y=hardware_peak,
            linestyle='--',
            linewidth=2.5,
            color='k',
            alpha=0.7,
            label=f'Hardware Peak ({hardware_peak:.0f} GFLOPS)'
        )

    # Axis labels, title, scales
    ax.set_xlabel('Matrix Size (M = N = K)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (GFLOPS)', fontsize=14, fontweight='bold')
    ax.set_title('GEMM Performance vs Matrix Size with Hardware Peak', fontsize=16, fontweight='bold', pad=20)

    # Use log scale on X (sizes) to better spread typical 128–8192 ranges
    ax.set_xscale('log', base=2)

    # Y-axis limits
    y_min = max(1, square_df['GFLOPS'].min() * 0.7)
    y_max_data = square_df['GFLOPS'].max()
    y_max = max(y_max_data, hardware_peak) * 1.2
    ax.set_ylim(y_min, y_max)

    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Roofline plot (size vs GFLOPS) saved to {output_file}")
    plt.close()

def plot_performance_comparison(df, output_file):
    """Plot performance comparison across different matrix sizes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Filter square matrices
    square_df = df[(df['M'] == df['N']) & (df['N'] == df['K'])].copy()
    
    if len(square_df) > 0:
        kernels = square_df['Kernel'].unique()
        colors = {
            'Lab1_Tiled': '#e74c3c',
            'cuBLAS_SGEMM': '#27ae60',
            'cuBLAS_HGEMM_TensorCore': '#3498db'
        }
        
        # Plot 1: GFLOPS vs Matrix Size
        for kernel in kernels:
            kernel_df = square_df[square_df['Kernel'] == kernel].sort_values('M')
            ax1.plot(kernel_df['M'], kernel_df['GFLOPS'],
                    marker='o', label=kernel, color=colors.get(kernel, 'gray'),
                    linewidth=2.5, markersize=8, markeredgewidth=1.5, markeredgecolor='white')
        
        ax1.set_xlabel('Matrix Size (M=N=K)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Performance (GFLOPS)', fontsize=13, fontweight='bold')
        ax1.set_title('GEMM Performance vs Matrix Size', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        
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
                
                ax2.plot(sizes, efficiency, marker='o', label=f"{kernel} vs cuBLAS",
                        color=colors.get(kernel, 'gray'), linewidth=2.5, markersize=8,
                        markeredgewidth=1.5, markeredgecolor='white')
        
        ax2.set_xlabel('Matrix Size (M=N=K)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Efficiency (% of cuBLAS FP32)', fontsize=13, fontweight='bold')
        ax2.set_title('Relative Efficiency Analysis', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xscale('log', base=2)
        ax2.axhline(y=100, color='#27ae60', linestyle='--', alpha=0.5, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Performance comparison saved to {output_file}")
    plt.close()

def generate_analysis_report(df, gpu_specs, output_file):
    """Generate detailed analysis report"""
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
    plot_roofline(df, gpu_specs, 'results/performance_plot.png')
    plot_performance_comparison(df, 'results/performance_comparison.png')
    generate_analysis_report(df, gpu_specs, 'results/analysis_report.txt')
    
    print("\nAnalysis complete!")
    print("Generated files:")
    print("  - results/performance_plot.png")
    print("  - results/performance_comparison.png")
    print("  - results/analysis_report.txt")

if __name__ == '__main__':
    main()