# Benchmark Results Summary

Run date: 2026-09-27

| Implementation | Training | Throughput | Inference | Fwd+Bwd | Train Acc | Unseen Acc |
|---|---|---|---|---|---|---|
| F# .NET 8 | 1.45s | 198,881 ex/s | 1.43 µs | 4.07 µs | 96/96 | 1/8 |
| C# .NET 8 | 2.25s | 127,879 ex/s | 2.02 µs | 6.49 µs | 96/96 | 0/8 |
| C# Web SDK | 2.22s | 129,767 ex/s | 2.14 µs | 6.36 µs | 96/96 | 0/8 |
| Rust release | 2.76s | 104,466 ex/s | 2.74 µs | 7.89 µs | 96/96 | 0/8 |
| Python float | 392s | 1,516 ex/s | 147.6 µs | 512.8 µs | 196/198 | — |
| Python NAND | 43.6s | ~37 ex/s | — | — | 30% exact | 83.8% bit |
