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

## Cross-language run — identical data, identical init (2026-09-29)

`bench/run.sh` trains every runtime on the same `bench/shared/corpus.txt` (96 words × 23 flags) from the same
seed-42 `bench/shared/init.txt`: online SGD, lr 0.5, 3001 epochs, embed 16 → tanh 32 → sigmoid 23.
Every runtime reaches the same final loss (9.1009e-5), so the runs are computing the same thing.
Best of 3, single thread, 4-core x86-64 Linux container.

| Rank | Runtime | Time | ex/s | Final loss |
|---|---|---|---|---|
| 1 | Rust, allocation-free (`src/rust/src/bin/lean.rs`) | 0.62 s | 463,000 | 9.1009e-5 |
| 2 | Mojo 1.1.0 `-O3` (`mojo/morphology.mojo`) | 0.91 s | 315,000 | 9.1009e-5 |
| 3 | Rust, original engine (`src/rust/src/bin/export.rs`) | 2.58 s | 112,000 | 9.1009e-5 |
| 4 | BQN, CBQN (`bqn/morphology.bqn`) | 2.77 s | 104,000 | 9.1009e-5 |
| 5 | Dyalog APL 20.0 (`dyalog-apl/morphology.apls`) | 5.91 s | 48,800 | 9.1009e-5 |
| 6 | Forth, gforth 0.7.3 (`forth/morphology.fs`) | 24.9 s | 11,600 | 9.1009e-5 |

Notes: the original Rust engine allocates `Vec`s in every forward and backward call; the lean port removes that
and is the fair native baseline. Dyalog and gforth are interpreters; APL's cost is per-primitive overhead on
tiny arrays (16–32 wide), where the vector primitives don't get room to pay off.
