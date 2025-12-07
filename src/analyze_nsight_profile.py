#!/usr/bin/env python3
"""
Nsight Compute Profile Analyzer
Parses and summarizes profiling results
"""

import subprocess
import sys
from pathlib import Path

# Try to import pandas, but make it optional
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Using basic analysis mode.")

def parse_ncu_report(ncu_file):
    """Parse Nsight Compute report file using ncu CLI"""
    try:
        # Use ncu CLI to export metrics - try summary page first (more reliable)
        result = subprocess.run(
            ['ncu', '--import', str(ncu_file), '--page', 'summary', '--print', 'gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.bytes_per_second.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_active'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            # Try alternative: export to CSV
            result = subprocess.run(
                ['ncu', '--import', str(ncu_file), '--page', 'summary', '--csv'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"Warning: Could not parse {ncu_file}")
            print(f"Error: {result.stderr}")
            return None
        
        # Parse output - look for key metrics
        data = {}
        lines = result.stdout.strip().split('\n')
        
        # Try to extract metrics from output
        for line in lines:
            # Look for common metric patterns
            if 'throughput' in line.lower() or 'occupancy' in line.lower() or 'duration' in line.lower():
                # Simple parsing - adjust based on actual ncu output format
                parts = line.split()
            if len(parts) >= 2:
                    try:
                        # Try to extract numeric values
                        for part in parts:
                            if '%' in part:
                                val = float(part.replace('%', ''))
                                if 'throughput' in line.lower() and 'compute' in line.lower():
                                    data['compute_throughput'] = val
                                elif 'throughput' in line.lower() and 'memory' in line.lower():
                                    data['memory_throughput'] = val
                            elif part.replace('.', '').isdigit():
                                val = float(part)
                                if 'occupancy' in line.lower() or 'active' in line.lower():
                                    data['occupancy'] = val
                    except:
                        pass
        
        return data if data else None
    except Exception as e:
        print(f"Error parsing {ncu_file}: {e}")
        return None

def analyze_kernel_metrics(profile_data, matrix_size):
    """Analyze key kernel performance metrics"""
    
    metrics = {
        'matrix_size': matrix_size,
        'occupancy': 0.0,
        'memory_throughput': 0.0,
        'compute_throughput': 0.0,
        'warp_stalls': {},
        'bank_conflicts': 0,
        'registers_per_thread': 0,
        'shared_mem_usage': 0
    }
    
    if not profile_data:
        return metrics
    
    # Extract key metrics (adjust keys based on actual ncu output)
    key_mappings = {
        'Occupancy': 'occupancy',
        'Achieved Occupancy': 'occupancy',
        'Memory Throughput': 'memory_throughput',
        'DRAM Throughput': 'memory_throughput',
        'Compute (SM) Throughput': 'compute_throughput',
        'SM Busy': 'compute_throughput',
        'Registers Per Thread': 'registers_per_thread',
        'Shared Memory': 'shared_mem_usage',
        'Bank Conflicts': 'bank_conflicts'
    }
    
    for key, value in profile_data.items():
        for pattern, metric in key_mappings.items():
            if pattern.lower() in key.lower():
                try:
                    # Extract numeric value
                    if '%' in value:
                        metrics[metric] = float(value.strip('%'))
                    else:
                        val_str = ''.join(c for c in value if c.isdigit() or c == '.')
                        if val_str:
                            metrics[metric] = float(val_str)
                except:
                    pass
    
    return metrics

def generate_profiling_report(profile_dir, output_file):
    """Generate comprehensive profiling report"""
    
    profile_dir = Path(profile_dir)
    ncu_files = list(profile_dir.glob('lab1_*.ncu-rep'))
    
    if not ncu_files:
        print(f"No .ncu-rep files found in {profile_dir}")
        return
    
    print(f"\nFound {len(ncu_files)} profile files")
    
    all_metrics = []
    
    for ncu_file in sorted(ncu_files):
        # Extract matrix size from filename
        size_str = ncu_file.stem.split('_')[-1]
        matrix_size = f"{size_str}x{size_str}x{size_str}"
        
        print(f"Processing {ncu_file.name}...")
        
        profile_data = parse_ncu_report(ncu_file)
        metrics = analyze_kernel_metrics(profile_data, matrix_size)
        all_metrics.append(metrics)
    
    # Write report
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NSIGHT COMPUTE PROFILING REPORT\n")
        f.write("Lab-1 Tiled GEMM Kernel Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        for metrics in all_metrics:
            f.write(f"\nMatrix Size: {metrics['matrix_size']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Occupancy: {metrics['occupancy']:.1f}%\n")
            f.write(f"Memory Throughput: {metrics['memory_throughput']:.1f}%\n")
            f.write(f"Compute Throughput: {metrics['compute_throughput']:.1f}%\n")
            
            if metrics['registers_per_thread'] > 0:
                f.write(f"Registers per Thread: {metrics['registers_per_thread']}\n")
            
            if metrics['shared_mem_usage'] > 0:
                f.write(f"Shared Memory Usage: {metrics['shared_mem_usage']} bytes\n")
            
            if metrics['bank_conflicts'] > 0:
                f.write(f"Bank Conflicts: {metrics['bank_conflicts']}\n")
            
            f.write("\n")
        
        # Summary and recommendations
        f.write("=" * 80 + "\n")
        f.write("ANALYSIS AND RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        avg_occupancy = sum(m['occupancy'] for m in all_metrics) / len(all_metrics)
        avg_mem_throughput = sum(m['memory_throughput'] for m in all_metrics) / len(all_metrics)
        avg_compute = sum(m['compute_throughput'] for m in all_metrics) / len(all_metrics)
        
        f.write(f"Average Occupancy: {avg_occupancy:.1f}%\n")
        f.write(f"Average Memory Throughput: {avg_mem_throughput:.1f}%\n")
        f.write(f"Average Compute Throughput: {avg_compute:.1f}%\n\n")
        
        f.write("Bottleneck Analysis:\n")
        if avg_occupancy < 50:
            f.write("  Warning: Low occupancy detected\n")
            f.write("    → Reduce register usage or shared memory per block\n")
            f.write("    → Increase threads per block\n\n")
        
        if avg_mem_throughput < 50:
            f.write("  Warning: Low memory throughput\n")
            f.write("    → Improve memory coalescing\n")
            f.write("    → Reduce global memory accesses via shared memory\n\n")
        
        if avg_compute < 50:
            f.write("  Warning: Low compute throughput\n")
            f.write("    → Increase arithmetic intensity\n")
            f.write("    → Better utilize FMA units\n\n")
        
        # Specific recommendations
        f.write("\nOptimization Priorities:\n")
        issues = []
        
        if avg_occupancy < 50:
            issues.append(("Occupancy", 1))
        if avg_mem_throughput < 50:
            issues.append(("Memory Throughput", 2))
        if avg_compute < 50:
            issues.append(("Compute Utilization", 3))
        
        if not issues:
            f.write("  Kernel shows good overall performance\n")
            f.write("  • Consider advanced optimizations (vectorization, async copies)\n")
        else:
            for issue, priority in sorted(issues, key=lambda x: x[1]):
                f.write(f"  {priority}. Address {issue}\n")
    
    print(f"Profiling report saved to {output_file}")

def main():
    print("\n" + "=" * 60)
    print("Nsight Compute Profile Analyzer")
    print("=" * 60 + "\n")
    
    profile_dir = Path("results/profiling")
    
    if not profile_dir.exists():
        print(f"Error: {profile_dir} does not exist")
        print("Please run the profiling script first:")
        print("  sbatch scripts/run_nsight_profile.sbatch")
        sys.exit(1)
    
    output_file = profile_dir / "profiling_summary.txt"
    generate_profiling_report(profile_dir, output_file)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print(f"\nReport: {output_file}")
    print("\nFor detailed analysis, open in Nsight Compute UI:")
    print("  ncu-ui results/profiling/lab1_*.ncu-rep\n")

if __name__ == '__main__':
    main()