import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./results_1.2/cublas_results.csv")
df_fp32 = df[df["dtype"] == "fp32"]
df_tc = df[df["dtype"].str.contains("f16")]

plt.figure(figsize=(8,6))
plt.plot(df_fp32["M"], df_fp32["TFLOPS"], 'o-', label="cuBLAS FP32")
plt.plot(df_tc["M"], df_tc["TFLOPS"], '^-', label="cuBLAS FP16→FP32 (TensorCore)")
plt.xlabel("Matrix Dimension (M=N=K)")
plt.ylabel("Throughput (TFLOPS)")
plt.title("Roofline-style Performance (cuBLAS Baselines, RTX 8000)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roofline_plot_fixed.png", dpi=200)