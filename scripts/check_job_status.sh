#!/bin/bash
# Quick script to check job status

echo "Current Jobs:"
squeue -u $USER

echo ""
echo "Recent Job Output:"
if [ -f logs/profile_comparison.out ]; then
    echo "Last 30 lines of profile_comparison.out:"
    tail -30 logs/profile_comparison.out
else
    echo "No profile_comparison.out found yet"
fi

echo ""
echo "Check if benchmark is running:"
if pgrep -f benchmark_gemm > /dev/null; then
    echo "benchmark_gemm process is running"
else
    echo "No benchmark_gemm process found"
fi

