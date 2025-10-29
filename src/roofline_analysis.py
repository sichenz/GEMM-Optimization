import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

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
    
    GEMM operations: 2*M*N*K
    Memory: M*K (read A) + K*N (read B) + M*N (write C) elements
    """
    flops = 2.0 * M * N * K
    bytes_transferred = (M * K + K * N + M * N) * bytes_per_element
    return flops / bytes_transferred

def plot_roofline(df, gpu_specs, output_file):
    """
    Create roofline plot
    
    Args:
        df: DataFrame with benchmark results
        gpu_specs: dict with 'peak_gflops_fp32', 'peak_gflops_fp16', 'peak_bandwidth_gb_s'
        output_file: path to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate arithmetic intensity for each benchmark
    df['bytes_per_element'] = df['DType'].apply(lambda x: 2 if x == 'FP16' else 4)
    df['AI'] = df.apply(lambda row: calculate_arithmetic_intensity(
        row['M'], row['N'], row['K'], row['bytes_per_element']), axis=1)
    
    # Roofline parameters
    peak_bandwidth = gpu_specs['peak_bandwidth_gb_s']
    peak_gflops_fp32 = gpu_specs['peak_gflops_fp32']
    peak_gflops_fp16 = gpu_specs['peak_gflops_fp16']
    
    # Create roofline curves
    ai_range = np.logspace(-1, 3, 1000)  # Arithmetic intensity from 0.1 to 1000
    
    # FP32 roofline
    memory_bound_fp32 = ai_range * peak_bandwidth
    compute_bound_fp32 = np.full_like(ai_range, peak_gflops_fp32)
    roofline_fp32 = np.minimum(memory_bound_fp32, compute_bound_fp32)
    
    # FP16 roofline (TensorCore)
    memory_bound_fp16 = ai_range * peak_bandwidth
    compute_bound_fp16 = np.full_like(ai_range, peak_gflops_fp16)
    roofline_fp16 = np.minimum(memory_bound_fp16, compute_bound_fp16)
    
    # Plot rooflines
    ax.loglog(ai_range, roofline_fp32, 'k-', linewidth=2, label='FP32 Roofline')
    ax.loglog(ai_range, roofline_fp16, 'b--', linewidth=2, label='FP16 TensorCore Roofline')
    
    # Plot benchmark results
    kernels = df['Kernel'].unique()
    colors = {'Lab1_Tiled': 'red', 'cuBLAS_SGEMM': 'green', 'cuBLAS_HGEMM_TensorCore': 'blue'}
    markers = {'Lab1_Tiled': 'o', 'cuBLAS_SGEMM': 's', 'cuBLAS_HGEMM_TensorCore': '^'}
    
    for kernel in kernels:
        kernel_df = df[df['Kernel'] == kernel]
        color = colors.get(kernel, 'gray')
        marker = markers.get(kernel, 'x')
        ax.loglog(kernel_df['AI'], kernel_df['GFLOPS'], 
                 marker=marker, color=color, markersize=8, 
                 linestyle='', label=kernel, alpha=0.7)
    
    # Labels and formatting
    ax.set_xlabel('Arithmetic Intensity (FLOPS/Byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax.set_title('Roofline Model: GEMM Performance Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)
    
    # Add annotations for peak values
    ax.axhline(y=peak_gflops_fp32, color='k', linestyle=':', alpha=0.3)
    ax.text(ai_range[-1], peak_gflops_fp32 * 1.1, 
            f'Peak FP32: {peak_gflops_fp32:.1f} GFLOPS', 
            ha='right', fontsize=9)
    
    if peak_gflops_fp16 > 0:
        ax.axhline(y=peak_gflops_fp16, color='b', linestyle=':', alpha=0.3)
        ax.text(ai_range[-1], peak_gflops_fp16 * 1.1, 
                f'Peak FP16 TC: {peak_gflops_fp16:.1f} GFLOPS', 
                ha='right', fontsize=9, color='blue')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Roofline plot saved to {output_file}")
    plt.close()

def plot_performance_comparison(df, output_file):
    """Plot performance comparison across different matrix sizes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter square matrices for cleaner visualization
    square_df = df[(df['M'] == df['N']) & (df['N'] == df['K'])]
    
    if len(square_df) > 0:
        # Plot 1: GFLOPS vs Matrix Size
        kernels = square_df['Kernel'].unique()
        colors = {'Lab1_Tiled': 'red', 'cuBLAS_SGEMM': 'green', 'cuBLAS_HGEMM_TensorCore': 'blue'}
        
        for kernel in kernels:
            kernel_df = square_df[square_df['Kernel'] == kernel].sort_values('M')
            ax1.plot(kernel_df['M'], kernel_df['GFLOPS'], 
                    marker='o', label=kernel, color=colors.get(kernel, 'gray'), linewidth=2)
        
        ax1.set_xlabel('Matrix Size (M=N=K)', fontsize=12)
        ax1.set_ylabel('Performance (GFLOPS)', fontsize=12)
        ax1.set_title('GEMM Performance vs Matrix Size', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        # Plot 2: Efficiency vs Matrix Size
        # Calculate efficiency relative to cuBLAS_SGEMM for FP32
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
                        color=colors.get(kernel, 'gray'), linewidth=2)
        
        ax2.set_xlabel('Matrix Size (M=N=K)', fontsize=12)
        ax2.set_ylabel('Efficiency (% of cuBLAS FP32)', fontsize=12)
        ax2.set_title('Relative Efficiency Analysis', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        ax2.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='cuBLAS baseline')
    
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
            efficiency = (kernel_df['GFLOPS'].mean() / peak) * 100
            f.write(f"  Average Efficiency: {efficiency:.2f}% of peak\n")
        
        # Comparison between Lab-1 and cuBLAS
        f.write("\n" + "=" * 80 + "\n")
        f.write("LAB-1 vs cuBLAS COMPARISON:\n")
        f.write("-" * 40 + "\n")
        
        lab1_df = df[df['Kernel'] == 'Lab1_Tiled']
        cublas_df = df[df['Kernel'] == 'cuBLAS_SGEMM']
        
        if len(lab1_df) > 0 and len(cublas_df) > 0:
            for _, lab1_row in lab1_df.iterrows():
                cublas_row = cublas_df[(cublas_df['M'] == lab1_row['M']) & 
                                       (cublas_df['N'] == lab1_row['N']) & 
                                       (cublas_df['K'] == lab1_row['K'])]
                if len(cublas_row) > 0:
                    speedup = cublas_row['GFLOPS'].values[0] / lab1_row['GFLOPS']
                    efficiency = (lab1_row['GFLOPS'] / cublas_row['GFLOPS'].values[0]) * 100
                    f.write(f"\nMatrix {lab1_row['M']}x{lab1_row['N']}x{lab1_row['K']}:\n")
                    f.write(f"  Lab-1: {lab1_row['GFLOPS']:.2f} GFLOPS\n")
                    f.write(f"  cuBLAS: {cublas_row['GFLOPS'].values[0]:.2f} GFLOPS\n")
                    f.write(f"  cuBLAS Speedup: {speedup:.2f}x\n")
                    f.write(f"  Lab-1 Efficiency: {efficiency:.2f}% of cuBLAS\n")
        
        # TensorCore analysis
        tensorcore_df = df[df['Kernel'] == 'cuBLAS_HGEMM_TensorCore']
        if len(tensorcore_df) > 0 and len(cublas_df) > 0:
            f.write("\n" + "=" * 80 + "\n")
            f.write("TENSORCORE ANALYSIS (FP16 vs FP32):\n")
            f.write("-" * 40 + "\n")
            
            for _, tc_row in tensorcore_df.iterrows():
                fp32_row = cublas_df[(cublas_df['M'] == tc_row['M']) & 
                                     (cublas_df['N'] == tc_row['N']) & 
                                     (cublas_df['K'] == tc_row['K'])]
                if len(fp32_row) > 0:
                    speedup = tc_row['GFLOPS'] / fp32_row['GFLOPS'].values[0]
                    f.write(f"\nMatrix {tc_row['M']}x{tc_row['N']}x{tc_row['K']}:\n")
                    f.write(f"  FP32: {fp32_row['GFLOPS'].values[0]:.2f} GFLOPS\n")
                    f.write(f"  FP16 TensorCore: {tc_row['GFLOPS']:.2f} GFLOPS\n")
                    f.write(f"  TensorCore Speedup: {speedup:.2f}x\n")
        
        # Optimization opportunities
        f.write("\n" + "=" * 80 + "\n")
        f.write("OPTIMIZATION OPPORTUNITIES:\n")
        f.write("-" * 40 + "\n")
        
        if len(lab1_df) > 0:
            avg_lab1_gflops = lab1_df['GFLOPS'].mean()
            avg_cublas_gflops = cublas_df['GFLOPS'].mean()
            performance_gap = avg_cublas_gflops - avg_lab1_gflops
            
            f.write(f"\n1. Performance Gap:\n")
            f.write(f"   Current Lab-1 average: {avg_lab1_gflops:.2f} GFLOPS\n")
            f.write(f"   cuBLAS average: {avg_cublas_gflops:.2f} GFLOPS\n")
            f.write(f"   Gap to close: {performance_gap:.2f} GFLOPS ({(performance_gap/avg_cublas_gflops)*100:.1f}%)\n")
            
            f.write(f"\n2. TensorCore Opportunity:\n")
            if len(tensorcore_df) > 0:
                avg_tc_gflops = tensorcore_df['GFLOPS'].mean()
                tc_speedup = avg_tc_gflops / avg_lab1_gflops
                f.write(f"   Potential speedup with TensorCores: {tc_speedup:.2f}x\n")
                f.write(f"   Potential performance: {avg_tc_gflops:.2f} GFLOPS\n")
            
            f.write(f"\n3. Memory Efficiency:\n")
            avg_bandwidth = lab1_df['Bandwidth_GB_s'].mean()
            bandwidth_efficiency = (avg_bandwidth / gpu_specs['peak_bandwidth_gb_s']) * 100
            f.write(f"   Current bandwidth utilization: {bandwidth_efficiency:.1f}%\n")
            if bandwidth_efficiency < 50:
                f.write(f"   → Focus on memory access patterns and coalescing\n")
            
            f.write(f"\n4. Arithmetic Intensity:\n")
            avg_ai = lab1_df['AI'].mean()
            f.write(f"   Average AI: {avg_ai:.2f} FLOPS/Byte\n")
            ridge_point = gpu_specs['peak_gflops_fp32'] / gpu_specs['peak_bandwidth_gb_s']
            f.write(f"   Ridge point (compute-bound threshold): {ridge_point:.2f} FLOPS/Byte\n")
            if avg_ai < ridge_point:
                f.write(f"   → Memory-bound: optimize memory access patterns\n")
            else:
                f.write(f"   → Compute-bound: optimize computation efficiency\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Analysis report saved to {output_file}")

def main():
    # GPU specifications (modify based on your GPU)
    # These will be overridden by actual measurements if gpu_specs.txt exists
    gpu_specs = {
        'peak_gflops_fp32': 8.1,      # T4 spec
        'peak_gflops_fp16': 65.0,     # T4 TensorCore spec
        'peak_bandwidth_gb_s': 320.0  # T4 spec
    }
    
    # Try to load actual GPU specs from file
    try:
        with open('results/gpu_specs.txt', 'r') as f:
            content = f.read()
            # Parse peak values from file (this is simplified, adjust as needed)
            for line in content.split('\n'):
                if 'Estimated FP32 Peak TFLOPS:' in line:
                    val = float(line.split(':')[1].strip().split()[0])
                    gpu_specs['peak_gflops_fp32'] = val * 1000
                elif 'TensorCore FP16 Peak TFLOPS:' in line:
                    val = float(line.split(':')[1].strip().split()[0])
                    gpu_specs['peak_gflops_fp16'] = val * 1000
                elif 'Peak Memory Bandwidth:' in line:
                    val = float(line.split(':')[1].strip().split()[0])
                    gpu_specs['peak_bandwidth_gb_s'] = val
    except:
        print("Using default T4 GPU specifications")
    
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