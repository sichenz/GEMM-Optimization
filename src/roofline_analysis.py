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
    Create roofline plot with improved visualization
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Calculate arithmetic intensity for each benchmark
    df['bytes_per_element'] = df['DType'].apply(lambda x: 2 if x == 'FP16' else 4)
    df['AI'] = df.apply(lambda row: calculate_arithmetic_intensity(
        row['M'], row['N'], row['K'], row['bytes_per_element']), axis=1)
    
    # Roofline parameters
    peak_bandwidth = gpu_specs['peak_bandwidth_gb_s']
    peak_gflops_fp32 = gpu_specs['peak_gflops_fp32']
    peak_gflops_fp16 = gpu_specs['peak_gflops_fp16']
    
    # Create roofline curves with better range
    ai_min = max(0.1, df['AI'].min() * 0.5)
    ai_max = min(1000, df['AI'].max() * 2)
    ai_range = np.logspace(np.log10(ai_min), np.log10(ai_max), 1000)
    
    # FP32 roofline
    memory_bound_fp32 = ai_range * peak_bandwidth
    compute_bound_fp32 = np.full_like(ai_range, peak_gflops_fp32)
    roofline_fp32 = np.minimum(memory_bound_fp32, compute_bound_fp32)
    
    # FP16 roofline (TensorCore)
    memory_bound_fp16 = ai_range * peak_bandwidth
    compute_bound_fp16 = np.full_like(ai_range, peak_gflops_fp16)
    roofline_fp16 = np.minimum(memory_bound_fp16, compute_bound_fp16)
    
    # Plot rooflines with thicker lines
    ax.loglog(ai_range, roofline_fp32, 'k-', linewidth=2.5, label='FP32 Roofline', zorder=1)
    ax.loglog(ai_range, roofline_fp16, 'b--', linewidth=2.5, label='FP16 TensorCore Roofline', zorder=1)
    
    # Plot benchmark results with distinct markers
    kernels = df['Kernel'].unique()
    colors = {
        'Lab1_Tiled': '#e74c3c',           # Red
        'cuBLAS_SGEMM': '#27ae60',         # Green
        'cuBLAS_HGEMM_TensorCore': '#3498db'  # Blue
    }
    markers = {
        'Lab1_Tiled': 'o', 
        'cuBLAS_SGEMM': 's', 
        'cuBLAS_HGEMM_TensorCore': '^'
    }
    
    for kernel in kernels:
        kernel_df = df[df['Kernel'] == kernel]
        color = colors.get(kernel, 'gray')
        marker = markers.get(kernel, 'x')
        ax.loglog(kernel_df['AI'], kernel_df['GFLOPS'], 
                 marker=marker, color=color, markersize=10, 
                 linestyle='', label=kernel, alpha=0.8, markeredgewidth=1.5,
                 markeredgecolor='white', zorder=3)
    
    # Calculate and show ridge points
    ridge_point_fp32 = peak_gflops_fp32 / peak_bandwidth
    ridge_point_fp16 = peak_gflops_fp16 / peak_bandwidth
    
    # Add vertical lines at ridge points
    ax.axvline(x=ridge_point_fp32, color='k', linestyle=':', alpha=0.3, linewidth=1.5)
    ax.axvline(x=ridge_point_fp16, color='b', linestyle=':', alpha=0.3, linewidth=1.5)
    
    # Labels and formatting
    ax.set_xlabel('Arithmetic Intensity (FLOPS/Byte)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (GFLOPS)', fontsize=14, fontweight='bold')
    ax.set_title('Roofline Model: GEMM Performance Analysis\nQuadro RTX 8000', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, which='both', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # Add annotations for peak values and ridge points
    y_offset_fp32 = peak_gflops_fp32 * 1.15
    y_offset_fp16 = peak_gflops_fp16 * 1.15
    
    ax.text(ai_range[-1] * 0.8, y_offset_fp32, 
            f'Peak FP32: {peak_gflops_fp32:.1f} GFLOPS', 
            ha='right', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if peak_gflops_fp16 > 0:
        ax.text(ai_range[-1] * 0.8, y_offset_fp16, 
                f'Peak FP16 TC: {peak_gflops_fp16:.1f} GFLOPS', 
                ha='right', fontsize=10, color='#3498db',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add ridge point annotations
    ax.text(ridge_point_fp32 * 1.1, peak_gflops_fp32 * 0.1, 
            f'Ridge Point\n{ridge_point_fp32:.1f} FLOPs/Byte',
            fontsize=9, ha='left', va='bottom', color='gray',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Set reasonable axis limits
    y_min = max(1, df['GFLOPS'].min() * 0.5)
    y_max = min(peak_gflops_fp16 * 2, df['GFLOPS'].max() * 5)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(ai_min, ai_max)
    
    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.tick_params(axis='both', which='minor', labelsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Roofline plot saved to {output_file}")
    plt.close()

def plot_performance_comparison(df, output_file):
    """Plot performance comparison across different matrix sizes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Filter square matrices for cleaner visualization
    square_df = df[(df['M'] == df['N']) & (df['N'] == df['K'])].copy()
    
    if len(square_df) > 0:
        # Plot 1: GFLOPS vs Matrix Size
        kernels = square_df['Kernel'].unique()
        colors = {
            'Lab1_Tiled': '#e74c3c', 
            'cuBLAS_SGEMM': '#27ae60', 
            'cuBLAS_HGEMM_TensorCore': '#3498db'
        }
        
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
        ax1.tick_params(axis='both', which='major', labelsize=11)
        
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
        ax2.axhline(y=100, color='#27ae60', linestyle='--', alpha=0.5, 
                    linewidth=2, label='cuBLAS baseline')
        ax2.tick_params(axis='both', which='major', labelsize=11)
    
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
        
        # Add more analysis sections...
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Analysis report saved to {output_file}")

def main():
    # GPU specifications - using values from your log
    gpu_specs = {
        'peak_gflops_fp32': 8100.0,    # Using actual values from gpu_specs output
        'peak_gflops_fp16': 65000.0,   # TensorCore spec
        'peak_bandwidth_gb_s': 624.1   # From gpu_specs output
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