# Project RED — Reproducible Empirical Benchmark Report

- **PyTorch Version:** `2.14.1+cpu` | **Python:** `3.13.15` | **Seed:** `42`
- **Hardware / Backend:** `CPU (Intel64 Family 6 Model 197 Stepping 2, GenuineIntel)` (`CUDA: CPU-Simulated-Ceiling`)
- **Workload:** 60-step fine-tuning loop (`hidden_dim=512`), sequence-length spikes (`64 -> 192` tokens) at steps 15 & 35, and transient FP16 `NaN` gradient overflow at step 25.

## Comparative Results

| Metric | Baseline (Static BS=32) | Baseline (Static BS=16) | Project RED (Dynamic BS=32) |
| :--- | :--- | :--- | :--- |
| **Completed Steps** | `0/60` (Aborted) | `14/60` | **`60/60`** |
| **Unhandled OOM Crashes** | `1` | `1` | **`0`** |
| **Auto OOM Recoveries (DBC)** | `0` | `0` | **`3`** |
| **NaN Grad Overflows Skipped** | `0` (Weights Poisoned) | `0` (Weights Poisoned) | **`1`** |
| **Model Weights Finite at End** | `True` | `True` | **`True`** |
| **Peak Allocated Memory (MB)** | `64.0 MB` | `96.0 MB` | **`48.0 MB`** |
| **Peak Reserved Memory (MB)** | `80.0 MB` | `120.0 MB` | **`53.76 MB`** |
| **Throughput (steps/sec)** | `0.0` | `66.65` | **`29.14`** |
| **Final Training Loss** | `None` | `1.67403` | **`0.10334`** |
