import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# ---- 1. Load data ----
p11 = ROOT / "/Users/dtquynhanh/Documents/NYU/Big Data & ML Systems/GEMM-Optimization/results1_1"
p12 = ROOT / "results1_2"
pcut = ROOT / "results_cutlass"

# Lab-1 + basic cuBLAS summary from benchmark_gemm
df_bench = pd.read_csv(p11 / "benchmark_results.csv")

# cuBLAS detailed microbenchmarks
df_cublas = pd.read_csv(p12 / "cublas_results.csv")

# CUTLASS (big CSVs)
df_cut_fp32 = pd.read_csv(pcut / "cutlass_fp32.gemm.csv")
df_cut_tc   = pd.read_csv(pcut / "cutlass_f16tc.gemm.csv")

# ---- 2. Normalize into a single table ----
rows = []

# From benchmark_results: assume columns like Kernel, DType, M,N,K,GFLOPS
for _, r in df_bench.iterrows():
    rows.append(dict(
        impl=r["Kernel"],      # e.g. "Lab1_Tiled", "cuBLAS_SGEMM", "cuBLAS_HGEMM_TensorCore"
        precision=r["DType"],  # "FP32" / "FP16"
        M=r["M"], N=r["N"], K=r["K"],
        tflops=r["GFLOPS"] / 1000.0
    ))

# From cublas_results: already has TFLOPS
for _, r in df_cublas.iterrows():
    rows.append(dict(
        impl=r["api"],         # "sgemm" / "gemmex"
        precision=r["dtype"],  # "fp32" / "f16f32"
        M=r["M"], N=r["N"], K=r["K"],
        tflops=r["TFLOPS"]
    ))

# From CUTLASS: filter GEMM ops you care about (square sizes, best config, etc.)
def add_cutlass(df, tag):
    for _, r in df.iterrows():
        rows.append(dict(
            impl=f"CUTLASS_{tag}",
            precision=r["ElementCompute"],
            M=r["m"], N=r["n"], K=r["k"],
            tflops=r["GFLOPs"] / 1000.0
        ))

add_cutlass(df_cut_fp32, "FP32")
add_cutlass(df_cut_tc, "TC")

df_all = pd.DataFrame(rows)

# ---- 3. Compute efficiency ----
PEAK_FP32 = 8.1     # TFLOPS
PEAK_TC   = 65.0    # TFLOPS

def eff(row):
    if "TC" in row["impl"] or "HGEMM" in row["impl"] or "f16" in str(row["precision"]).lower():
        peak = PEAK_TC
    else:
        peak = PEAK_FP32
    return 100.0 * row["tflops"] / peak

df_all["efficiency_pct"] = df_all.apply(eff, axis=1)

# ---- 4. Example plot: FP32 square sizes ----
sq = df_all[(df_all["M"] == df_all["N"]) & (df_all["M"] == df_all["K"])]

fp32 = sq[df_all["precision"].astype(str).str.contains("32") &
          ~sq["impl"].str.contains("TC")]

plt.figure()
for impl, g in fp32.groupby("impl"):
    g_sorted = g.sort_values("M")
    plt.plot(g_sorted["M"], g_sorted["tflops"], marker="o", label=impl)

plt.xlabel("Matrix size (N, where M=N=K)")
plt.ylabel("Throughput (TFLOPS)")
plt.title("FP32 GEMM Performance – Lab1 vs cuBLAS vs CUTLASS")
plt.legend()
plt.xscale("log", base=2)
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig(ROOT / "results_phase1/phase1_fp32_comparison.png", dpi=200)

# ---- 5. Save combined table for your report ----
outdir = ROOT / "results_phase1"
outdir.mkdir(exist_ok=True)
df_all.to_csv(outdir / "phase1_combined_results.csv", index=False)

print("Saved:")
print("  - results_phase1/phase1_combined_results.csv")
print("  - results_phase1/phase1_fp32_comparison.png")